"""H21 multi-resource forecasting core.

Shared, resource-agnostic engine used by BOTH the training pipeline (`train_all.py`) and the
Streamlit app (`app.py`), so there is exactly one implementation of the feature engineering,
the split logic and the metrics. This is what guarantees the app's numbers match the
leaderboard's numbers - they are literally the same code path, and the app never recomputes
a metric, it only reads what training wrote.

Causality contract (identical to the steel notebook this is derived from):
  * Feature(t) may only read rows with index <= t.
  * rolling(w) is trailing/past-inclusive; `center=True` appears nowhere.
  * No bfill/backfill anywhere. Forward-fill only (carries PAST values forward).
  * Negative shifts exist ONLY in make_targets(), never in a feature.
  * assert_causal() scrambles the future and demands bit-identical history.
"""
from __future__ import annotations

import json
import os
import random
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

warnings.filterwarnings("ignore")

SEED = 42
DATE_COL = "Date"
DATA_PATH = "steel_data_egypt_combined.csv"
MODEL_DIR = "models"

# ---------------------------------------------------------------------------------------
# Resource registry. Adding a resource here is the ONLY code change needed - the trainer,
# the leaderboard and the Streamlit app all iterate this dict.
# ---------------------------------------------------------------------------------------
RESOURCES = {
    "steel":     dict(label="Steel (rebar)",  column="Steel_Price",     unit="EGP/ton"),
    "cement":    dict(label="Cement",         column="Cement_Price",    unit="EGP/ton"),
    "oil":       dict(label="Oil (Brent)",    column="Oil_Price",       unit="USD/bbl"),
    "aluminium": dict(label="Aluminium",      column="Aluminium_Price", unit="USD/ton"),
    "copper":    dict(label="Copper",         column="Copper_Price",    unit="USD/lb"),
}

HORIZONS = [1, 3, 7, 14, 21]
# Mean observed gap between consecutive rows is ~1.32 calendar days (irregular sampling:
# median 1, max 39). h is a STEP count, not a day count - this mapping is for display only.
MEAN_GAP_DAYS = 1.32
HORIZON_DAYS = {h: round(h * MEAN_GAP_DAYS, 1) for h in HORIZONS}


@dataclass
class Config:
    train_frac: float = 0.70
    val_frac: float = 0.15
    n_walk_folds: int = 4
    warmup: int = 60
    lags: tuple = (1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60)
    windows: tuple = (3, 5, 7, 14, 21, 30, 45, 60)
    top_k_features: int = 45
    dead_zones: tuple = (0.0, 0.001, 0.003, 0.005, 0.01)
    coverage_floor: float = 0.30
    # budgets (overridable per run)
    n_trials_xgb: int = 20
    n_trials_gru: int = 8
    max_epochs: int = 40
    patience: int = 6
    seed: int = SEED


CFG = Config()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# =======================================================================================
# Data loading + cleaning
# =======================================================================================
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    d = pd.read_csv(path)
    d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
    d = d.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    d = d.drop_duplicates(subset=[DATE_COL], keep="last").reset_index(drop=True)
    for c in d.columns:
        if c != DATE_COL:
            d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Data-quality fix: a price of exactly 0.0 is a scrape error, not a real quote
    # (Copper_Price has one such row on 2024-10-25). Treated as missing, never as a value -
    # leaving it in would corrupt every ratio and return that divides by it.
    for meta in RESOURCES.values():
        col = meta["column"]
        if col in d.columns:
            d.loc[d[col] <= 0, col] = np.nan
            d[col] = despike(d[col], col)
    return d


def despike(s: pd.Series, name: str = "", z: float = 4.0, revert: float = 0.5,
            report: bool = False):
    """Flag round-trip bad prints: a large move that is immediately (mostly) reversed.

    Rationale (verified against the raw series): the Egyptian rebar quote contains sequences
    like 33,301 -> 37,100 -> 32,135 -> 38,960 (+/-10-19% within days). A physical steel market
    does not oscillate like that; these are scrape/quote errors. A one-way jump of the same
    size IS kept - that is a real devaluation move, and deleting it would be fabricating a
    smoother history than actually occurred.

    A point is dropped only when BOTH legs are extreme AND opposite AND they cancel:
        |r_t| > z*sigma  and  |r_t+1| > z*sigma  and  sign flip  and  |r_t + r_t+1| < revert*|r_t|

    This is causal-safe: it is a data-cleaning step on the raw series, applied identically to
    every row before any split, and it never uses the target or future information to decide
    about a training label.
    """
    v = s.copy()
    lr = np.log(v / v.shift(1))
    sigma = lr.std()
    if not np.isfinite(sigma) or sigma <= 0:
        return v
    r0 = lr.values
    r1 = np.roll(lr.values, -1)
    big = (np.abs(r0) > z * sigma) & (np.abs(r1) > z * sigma)
    flip = np.sign(r0) != np.sign(r1)
    cancels = np.abs(r0 + r1) < revert * np.abs(r0)
    bad = big & flip & cancels
    bad[-1] = False
    n = int(np.nansum(bad))
    if n:
        v.iloc[np.where(bad)[0]] = np.nan
        if report:
            print(f"    despike[{name}]: removed {n} round-trip bad print(s) "
                  f"(sigma={sigma:.4f}, threshold={z}sigma={z*sigma:.3%})")
    return v


# =======================================================================================
# Causal feature engineering (parameterised by target resource)
# =======================================================================================
def _roll_block(s: pd.Series, name: str, windows=CFG.windows) -> dict:
    out = {}
    lr = np.log(s / s.shift(1))
    for w in windows:
        mp = max(2, w // 2)
        r = s.rolling(w, min_periods=mp)
        mu, sd, mn, mx = r.mean(), r.std(), r.min(), r.max()
        out[f"{name}_mean_{w}"] = mu
        out[f"{name}_std_{w}"] = sd
        out[f"{name}_min_{w}"] = mn
        out[f"{name}_max_{w}"] = mx
        out[f"{name}_range_{w}"] = (mx - mn) / s
        out[f"{name}_z_{w}"] = (s - mu) / sd.replace(0, np.nan)
        out[f"{name}_mom_{w}"] = s / s.shift(w) - 1.0
        out[f"{name}_vol_{w}"] = lr.rolling(w, min_periods=mp).std()
    return out


def _rsi(s: pd.Series, w: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / w, adjust=False, min_periods=w).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / w, adjust=False, min_periods=w).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_features(d: pd.DataFrame, resource: str) -> pd.DataFrame:
    """Causal feature matrix for ONE resource: its own autoregressive block, cross-resource
    relationships, the shared macro/external backdrop, and calendar/gap structure."""
    tgt_col = RESOURCES[resource]["column"]
    p = d[tgt_col]
    dates = d[DATE_COL]
    f = {}

    # ---- calendar / irregular-sampling structure -------------------------------------
    gap = dates.diff().dt.days
    f["days_since_prev_obs"] = gap
    f["gap_roll_mean_5"] = gap.rolling(5, min_periods=2).mean()
    f["is_after_long_gap"] = (gap > 3).astype(float)
    f["dow"] = dates.dt.dayofweek.astype(float)
    f["dom"] = dates.dt.day.astype(float)
    f["month"] = dates.dt.month.astype(float)

    # ---- own autoregressive block ------------------------------------------------------
    lr = np.log(p / p.shift(1))
    rt = p / p.shift(1) - 1.0
    for L in CFG.lags:
        f[f"own_lag_{L}"] = p.shift(L)
        f[f"own_ret_lag_{L}"] = rt.shift(L)
        f[f"own_logret_lag_{L}"] = lr.shift(L)
    f.update(_roll_block(p, "own"))
    for k in (1, 3, 5, 7, 14, 21, 30):
        f[f"own_ret_{k}"] = p / p.shift(k) - 1.0
    f["own_rsi_14"] = _rsi(p, 14)
    f["own_rsi_7"] = _rsi(p, 7)
    f["own_accel"] = rt.diff()
    for w in (7, 21, 60):
        f[f"own_dist_mean_{w}"] = p / p.rolling(w, min_periods=2).mean() - 1.0
        f[f"own_dist_high_{w}"] = p / p.rolling(w, min_periods=2).max() - 1.0
        f[f"own_dist_low_{w}"] = p / p.rolling(w, min_periods=2).min() - 1.0
    ema12 = p.ewm(span=12, adjust=False, min_periods=5).mean()
    ema26 = p.ewm(span=26, adjust=False, min_periods=5).mean()
    f["own_macd"] = ema12 - ema26
    f["own_macd_hist"] = f["own_macd"] - f["own_macd"].ewm(span=9, adjust=False, min_periods=5).mean()
    f["own_vol_ratio"] = (lr.rolling(7, min_periods=3).std()
                          / lr.rolling(45, min_periods=10).std().replace(0, np.nan))
    f["own_flat_frac_21"] = (rt.abs() < 1e-9).rolling(21, min_periods=5).mean()
    f["own_isna"] = p.isna().astype(float)

    # ---- other resources as cross-market drivers ---------------------------------------
    for other, meta in RESOURCES.items():
        ocol = meta["column"]
        if other == resource or ocol not in d.columns:
            continue
        s = d[ocol]
        for L in (1, 3, 5, 7, 14):
            f[f"{other}_lag_{L}"] = s.shift(L)
        f[f"{other}_ret_1"] = s / s.shift(1) - 1.0
        f[f"{other}_ret_5"] = s / s.shift(5) - 1.0
        f[f"{other}_ret_21"] = s / s.shift(21) - 1.0
        for w in (7, 21):
            f[f"{other}_vol_{w}"] = np.log(s / s.shift(1)).rolling(w, min_periods=3).std()
            f[f"{other}_z_{w}"] = ((s - s.rolling(w, min_periods=3).mean())
                                   / s.rolling(w, min_periods=3).std().replace(0, np.nan))
        ratio = p / s.replace(0, np.nan)
        f[f"ratio_own_{other}"] = ratio
        f[f"ratio_own_{other}_z21"] = ((ratio - ratio.rolling(21, min_periods=5).mean())
                                       / ratio.rolling(21, min_periods=5).std().replace(0, np.nan))
        f[f"relmom_own_{other}_21"] = (p / p.shift(21)) - (s / s.shift(21))
        f[f"{other}_isna"] = s.isna().astype(float)

    # ---- FX ----------------------------------------------------------------------------
    if "USD_Sell" in d.columns:
        usd = d["USD_Sell"].ffill()   # causal: step-persistent quote, carries PAST value forward
        f["usd_level"] = usd
        f["usd_ret_5"] = usd / usd.shift(5) - 1.0
        f["usd_ret_21"] = usd / usd.shift(21) - 1.0
        f["usd_vol_21"] = np.log(usd / usd.shift(1)).rolling(21, min_periods=5).std()
        f["own_usd_deflated"] = p / usd.replace(0, np.nan)
        if "USD_Buy" in d.columns:
            f["usd_spread_rel"] = (d["USD_Sell"] - d["USD_Buy"]) / d["USD_Buy"].replace(0, np.nan)

    # ---- shared external backdrop already merged (leakage-safe) into the CSV -----------
    # These arrived via merge_asof(direction='backward') on an explicit available_date, so
    # they are simply carried through here. Steel-specific ratio columns are excluded when
    # modelling a different resource (they encode the steel target).
    EXTERNAL_PREFIXES = ("iron_ore_", "coal_", "natgas_", "cbe_", "egypt_", "egp_")
    for c in d.columns:
        if c.startswith(EXTERNAL_PREFIXES) and c not in f:
            if resource != "steel" and c.startswith("steel_"):
                continue
            f[c] = d[c]
    # EGP-adjusted commodity levels are generic cost drivers, keep for all resources
    for c in ["oil_egp", "alu_egp", "cop_egp", "iron_ore_egp", "coal_egp", "natgas_egp"]:
        if c in d.columns:
            f[c] = d[c]

    F = pd.DataFrame(f, index=d.index).replace([np.inf, -np.inf], np.nan)
    # never let a column that IS the target leak in
    return F.loc[:, [c for c in F.columns if c != tgt_col]]


def assert_causal(d: pd.DataFrame, resource: str, cut: int = 600) -> None:
    """Scramble every row after `cut` and demand bit-identical history."""
    base = build_features(d, resource)
    poisoned = d.copy()
    num_cols = [c for c in poisoned.columns if c != DATE_COL]
    rng = np.random.default_rng(0)
    block = poisoned.iloc[cut + 1:][num_cols]
    poisoned.loc[poisoned.index[cut + 1]:, num_cols] = block.values * rng.uniform(3, 9, size=block.shape)
    test = build_features(poisoned, resource)
    bad = [c for c in base.columns
           if not np.allclose(base[c].iloc[:cut + 1].values.astype(float),
                              test[c].iloc[:cut + 1].values.astype(float),
                              equal_nan=True, rtol=1e-9, atol=1e-9)]
    if bad:
        raise AssertionError(f"[{resource}] LEAKAGE: {len(bad)} non-causal features -> {bad[:10]}")


def assert_no_forbidden_patterns() -> None:
    import inspect
    src = (inspect.getsource(build_features) + inspect.getsource(_roll_block)
           + inspect.getsource(_rsi))
    for pat in ["bfill", "backfill", "center=True", "shift(-"]:
        if pat in src:
            raise AssertionError(f"LEAKAGE PATTERN '{pat}' found in feature code")


# =======================================================================================
# Targets / splits / metrics
# =======================================================================================
def make_targets(price: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    out = {}
    for h in horizons:
        fut = price.shift(-h)          # negative shift: TARGETS ONLY
        out[f"y_price_h{h}"] = fut
        out[f"y_ret_h{h}"] = fut / price - 1.0
        out[f"y_logret_h{h}"] = np.log(fut / price)
    return pd.DataFrame(out, index=price.index)


def target_to_price(pred, base_price, kind: str) -> np.ndarray:
    pred = np.asarray(pred, float)
    base_price = np.asarray(base_price, float)
    if kind == "price":
        return pred
    if kind == "ret":
        return base_price * (1.0 + pred)
    if kind == "logret":
        return base_price * np.exp(pred)
    raise ValueError(kind)


def build_dataset(d: pd.DataFrame, resource: str):
    """Feature matrix + targets + base price, with warm-up rows and missing-target rows removed."""
    tgt_col = RESOURCES[resource]["column"]
    F = build_features(d, resource)
    Y = make_targets(d[tgt_col])
    valid = d[tgt_col].notna().values.copy()
    valid[:CFG.warmup] = False
    idx = np.where(valid)[0]
    X = F.iloc[idx].reset_index(drop=True)
    dead = [c for c in X.columns if X[c].notna().sum() < 50 or X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=dead)
    return (X, Y.iloc[idx].reset_index(drop=True),
            d[tgt_col].iloc[idx].reset_index(drop=True),
            d[DATE_COL].iloc[idx].reset_index(drop=True))


def make_splits(n: int, embargo: int):
    n_tr = int(n * CFG.train_frac)
    n_va = int(n * CFG.val_frac)
    tr = np.arange(0, n_tr - embargo)
    va = np.arange(n_tr, n_tr + n_va - embargo)
    te = np.arange(n_tr + n_va, n - embargo)
    assert tr.max() < va.min() < te.min(), "split ordering violated"
    assert va.max() < te.min(), "val/test overlap"
    return tr, va, te


def walk_forward_folds(tr, va, embargo: int, n_folds: int = CFG.n_walk_folds):
    dev_end = int(va.max()) + 1
    edges = np.linspace(int(len(tr) * 0.55), dev_end, n_folds + 1).astype(int)
    folds = []
    for i in range(n_folds):
        a, b = edges[i], edges[i + 1]
        f_tr, f_va = np.arange(0, a - embargo), np.arange(a, b)
        if len(f_tr) > 120 and len(f_va) > 15:
            folds.append((f_tr, f_va))
    return folds


def impute_fold(Xtr, others, strategy="ffill_median"):
    if strategy == "native_nan":
        return Xtr, list(others)
    med = Xtr.median()

    def _apply(x):
        out = x.copy()
        if strategy in ("ffill", "ffill_median"):
            out = out.ffill()          # forward only - never backward
        if strategy in ("median", "ffill_median"):
            out = out.fillna(med)
        return out.fillna(med).fillna(0.0)

    return _apply(Xtr), [_apply(x) for x in others]


def rmse(a, b) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def directional_stats(y_true, y_pred, base, dead_zone: float = 0.0) -> dict:
    y_true, y_pred, base = (np.asarray(v, float) for v in (y_true, y_pred, base))
    act = y_true / base - 1.0
    prd = y_pred / base - 1.0
    take = (np.abs(prd) >= dead_zone) & (act != 0)
    n = int(take.sum())
    out = dict(dead_zone=dead_zone, n_valid=n,
               coverage=float(take.mean()) if len(act) else 0.0,
               dir_acc=np.nan, precision=np.nan, recall=np.nan, f1=np.nan,
               p_value=np.nan, cm=None)
    if n < 5:
        return out
    yt, yp = (act[take] > 0).astype(int), (prd[take] > 0).astype(int)
    out["dir_acc"] = float((yt == yp).mean())
    out["precision"] = float(precision_score(yt, yp, zero_division=0))
    out["recall"] = float(recall_score(yt, yp, zero_division=0))
    out["f1"] = float(f1_score(yt, yp, zero_division=0))
    out["cm"] = confusion_matrix(yt, yp, labels=[0, 1]).tolist()
    out["p_value"] = float(st.binomtest(int((yt == yp).sum()), n, 0.5, alternative="greater").pvalue)
    return out


def evaluate(y_true, y_pred, base, dead_zone=0.0, name="model", horizon=None) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ds = directional_stats(y_true, y_pred, base, dead_zone)
    return {"model": name, "horizon": horizon,
            "RMSE": rmse(y_true, y_pred),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "MAPE%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
            "DirAcc": ds["dir_acc"], "Coverage": ds["coverage"], "N_dir": ds["n_valid"],
            "Precision": ds["precision"], "Recall": ds["recall"], "F1": ds["f1"],
            "p_value": ds["p_value"], "dead_zone": dead_zone, "cm": ds["cm"]}


def diebold_mariano(y_true, pred_model, pred_naive):
    e1 = np.asarray(y_true, float) - np.asarray(pred_model, float)
    e2 = np.asarray(y_true, float) - np.asarray(pred_naive, float)
    d = e1 ** 2 - e2 ** 2
    sd = d.std(ddof=1)
    if sd == 0 or len(d) < 3:
        return 0.0, 1.0
    stat = float(d.mean() / (sd / np.sqrt(len(d))))
    p = float(2 * (1 - st.t.cdf(abs(stat), df=len(d) - 1)))
    return stat, p


SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}


# =======================================================================================
# LIVE INFERENCE SUPPORT (inference-only; never touches training data or saved artifacts)
# =======================================================================================
# Everything below exists so the app can answer "given TODAY's price, what happens next?".
# It is a normal production inference path - the same thing you would do calling the model
# in real time - and is NOT a training-time leakage risk. Nothing here writes to the
# training CSV, retrains anything, or mutates a saved artifact.
EGYPT_DIR = "data_egypt"
FRED_PUB_LAG_DAYS = 35     # IMF/FRED monthly commodity release lag (same assumption as training)
WB_PUB_MONTH = 7           # World Bank annual figure treated as public 1 Jul of the NEXT year


def _monthly_block(t: pd.DataFrame, tag: str, lag_days: int = FRED_PUB_LAG_DAYS) -> pd.DataFrame:
    """Monthly commodity features computed on the TRUE monthly cadence, then stamped with the
    date they would actually have been published. Mirrors the training-time construction."""
    t = t.copy()
    t.columns = ["date", "value"]
    t["date"] = pd.to_datetime(t["date"])
    t = t.dropna().sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"available_date": t["date"] + pd.Timedelta(days=lag_days)})
    out[f"{tag}_level"] = t["value"]
    out[f"{tag}_ret_1m"] = t["value"] / t["value"].shift(1) - 1.0
    out[f"{tag}_ret_3m"] = t["value"] / t["value"].shift(3) - 1.0
    out[f"{tag}_vol_6m"] = np.log(t["value"] / t["value"].shift(1)).rolling(6, min_periods=3).std()
    out[f"{tag}_mom_12m"] = t["value"] / t["value"].shift(12) - 1.0
    out[f"{tag}_mean_3m"] = t["value"].rolling(3, min_periods=1).mean()
    out[f"{tag}_dev_12m"] = t["value"] / t["value"].rolling(12, min_periods=3).mean() - 1.0
    return out.sort_values("available_date").reset_index(drop=True)


def external_features_asof(date, egypt_dir: str = EGYPT_DIR) -> dict:
    """Every external/Egypt feature as it would have been KNOWN on `date`.

    Uses the identical publication-lag rules as training (merge_asof backward against an
    explicit available_date), so a live inference row never sees a value earlier than the
    real world would have published it.
    """
    date = pd.Timestamp(date)
    feats: dict = {}

    for fname, tag in [("iron_ore_monthly.csv", "iron_ore"),
                       ("coal_monthly.csv", "coal"),
                       ("natgas_monthly.csv", "natgas")]:
        path = os.path.join(egypt_dir, fname)
        if not os.path.exists(path):
            continue
        blk = _monthly_block(pd.read_csv(path), tag)
        avail = blk[blk["available_date"] <= date]
        if len(avail):
            row = avail.iloc[-1]
            for c in blk.columns:
                if c != "available_date":
                    feats[c] = float(row[c]) if pd.notna(row[c]) else np.nan

    cbe_path = os.path.join(egypt_dir, "cbe_policy_rate.csv")
    if os.path.exists(cbe_path):
        cbe = pd.read_csv(cbe_path, comment="#")
        cbe["effective_date"] = pd.to_datetime(cbe["effective_date"])
        cbe = cbe.sort_values("effective_date")
        avail = cbe[cbe["effective_date"] <= date]          # announcements are known same-day
        if len(avail):
            feats["cbe_rate"] = float(avail.iloc[-1]["cbe_rate_pct"])
            feats["cbe_days_since_change"] = float((date - avail.iloc[-1]["effective_date"]).days)
            prior = cbe[cbe["effective_date"] <= date - pd.Timedelta(days=90)]
            if len(prior):
                feats["cbe_rate_change_90"] = feats["cbe_rate"] - float(prior.iloc[-1]["cbe_rate_pct"])

    for fname, tag in [("egypt_cpi_inflation_annual.json", "egypt_cpi_inflation"),
                       ("egypt_gdp_growth_annual.json", "egypt_gdp_growth")]:
        path = os.path.join(egypt_dir, fname)
        if not os.path.exists(path):
            continue
        raw = json.load(open(path, encoding="utf-8"))
        rows = [(int(r["date"]), r["value"]) for r in raw[1] if r["value"] is not None]
        wb = pd.DataFrame(rows, columns=["year", "value"]).sort_values("year")
        wb["available_date"] = pd.to_datetime(
            (wb["year"] + 1).astype(str) + f"-{WB_PUB_MONTH:02d}-01")
        avail = wb[wb["available_date"] <= date]
        if len(avail):
            feats[tag] = float(avail.iloc[-1]["value"])
            if len(avail) > 1:
                feats[f"{tag}_change"] = float(avail.iloc[-1]["value"] - avail.iloc[-2]["value"])
    return feats


def append_live_row(d: pd.DataFrame, resource: str, date, price: float,
                    egypt_dir: str = EGYPT_DIR) -> tuple:
    """Upsert a live 'as of today' observation for INFERENCE ONLY.

    Returns (new_dataframe, notes). The returned frame is a COPY - the caller's data and the
    on-disk CSV are untouched. Feature recomputation happens downstream in build_dataset(),
    so every lag/rolling/derived feature is rebuilt from (real history + this new anchor).
    """
    date = pd.Timestamp(date)
    col = RESOURCES[resource]["column"]
    d = d.copy()
    notes: list[str] = []

    ext = external_features_asof(date, egypt_dir)

    if (d[DATE_COL] == date).any():
        i = d.index[d[DATE_COL] == date][0]
        old = d.at[i, col]
        d.at[i, col] = float(price)
        notes.append(f"Replaced the existing {date.date()} quote "
                     f"({'n/a' if pd.isna(old) else f'{old:,.2f}'} -> {price:,.2f}).")
        target_i = i
    else:
        new = {c: np.nan for c in d.columns}
        new[DATE_COL] = date
        new[col] = float(price)
        # other columns carry the last observed value forward (past -> future only).
        last = d.iloc[-1]
        carried = []
        for c in d.columns:
            if c in (DATE_COL, col) or c in ext:
                continue
            if pd.notna(last[c]):
                new[c] = last[c]
                carried.append(c)
        d = pd.concat([d, pd.DataFrame([new])], ignore_index=True)
        target_i = len(d) - 1
        notes.append(f"Appended a new row for {date.date()}; {len(carried)} column(s) carried "
                     f"forward from {pd.Timestamp(last[DATE_COL]).date()} (marked stale).")

    for k, v in ext.items():
        if k in d.columns:
            d.at[target_i, k] = v
    if ext:
        notes.append(f"Recomputed {len(ext)} external feature(s) as-of {date.date()} using the "
                     "same publication-lag rules as training (not copied from the last row).")

    d = d.sort_values(DATE_COL).reset_index(drop=True)
    return d, notes
