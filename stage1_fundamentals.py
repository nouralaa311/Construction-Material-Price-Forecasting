"""STAGE 1 — Fix the fundamentals.

Runs on the resources Stage 0 flagged as having real structure (steel, cement) plus a control
from the no-signal group, so any "improvement" can be sanity-checked against a series where we
KNOW there is nothing to find. If a trick improves the no-signal control too, it is an artifact.

  1c  VALIDATION HYGIENE  - purge/embargo audit, asserted not eyeballed (run first: a leaking
                            split invalidates every later number).
  1a  TARGET DEFINITION   - price vs simple return vs log-return vs volatility-normalized return.
  1d  OUTLIERS & REGIME   - full history vs recency-weighted vs truncated-window training.
  1e  STATIONARITY        - fractional differencing of the target series.

Everything is scored by walk-forward validation in PRICE space, so all variants are comparable,
and always reported as a delta against the naive baseline.

Usage:  python stage1_fundamentals.py [--quick]
"""
from __future__ import annotations

import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

import h21_core as C
from train_all import fit_xgb, _INF

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

PROBE = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85,
             colsample_bytree=0.7, min_child_weight=4, reg_lambda=2.0)

SIGNAL_RESOURCES = ["steel", "cement"]
CONTROL_RESOURCE = "aluminium"          # Stage 0: no signal -> nothing should help here


# =======================================================================================
# 1c. Embargo / purge audit
# =======================================================================================
def audit_embargo(verbose: bool = True) -> bool:
    """A horizon-h label spans h rows into the future. If the last training row's label window
    reaches the validation block, labels overlap across the split and the fold leaks.
    Required: gap between train end and validation start >= h.
    """
    ok = True
    d = C.load_data()
    X, _, _, _ = C.build_dataset(d, "steel")
    emb = max(C.HORIZONS)
    tr, va, te = C.make_splits(len(X), emb)
    folds = C.walk_forward_folds(tr, va, emb)

    if verbose:
        print("=" * 96)
        print("1c. VALIDATION HYGIENE — PURGE / EMBARGO AUDIT".center(96))
        print("=" * 96)
        print(f"configured embargo = {emb} rows (= max horizon); folds = {len(folds)}")

    for i, (f_tr, f_va) in enumerate(folds, 1):
        gap = int(f_va.min() - f_tr.max() - 1)
        worst_h = max(C.HORIZONS)
        leaks = gap < worst_h
        ok &= not leaks
        if verbose:
            print(f"  fold {i}: train ends {f_tr.max():>4} | val starts {f_va.min():>4} | "
                  f"gap = {gap:>3} rows | need >= {worst_h} -> {'LEAK' if leaks else 'clean'}")
        # explicit label-overlap check: does any training label reach into validation?
        last_label_row = f_tr.max() + worst_h
        if last_label_row >= f_va.min():
            ok = False
            if verbose:
                print(f"        !! last train label lands at row {last_label_row} "
                      f">= val start {f_va.min()}")

    # train/val/test block boundaries
    for a, b, nm in ((tr, va, "train->val"), (va, te, "val->test")):
        gap = int(b.min() - a.max() - 1)
        bad = gap < max(C.HORIZONS)
        ok &= not bad
        if verbose:
            print(f"  {nm:<11} gap = {gap:>3} rows -> {'LEAK' if bad else 'clean'}")

    if verbose:
        print(f"\n  RESULT: {'PASS - no label overlap anywhere' if ok else 'FAIL - embargo too small'}")
    return ok


# =======================================================================================
# Target construction (1a) incl. volatility-normalized returns
# =======================================================================================
def make_target(Y: pd.DataFrame, P: pd.Series, h: int, kind: str, vol: pd.Series):
    """Return (y, inverse_fn). inverse_fn maps predictions back to PRICE space."""
    if kind in ("price", "ret", "logret"):
        y = Y[f"y_{kind}_h{h}"].values
        return y, (lambda pred, base, idx: C.target_to_price(pred, base, kind))
    if kind == "volnorm":
        # log-return scaled by trailing realized vol (causal). Equalizes the target's scale
        # across calm and turbulent regimes so the loss is not dominated by high-vol periods.
        v = vol.values
        y = Y[f"y_logret_h{h}"].values / np.where(v > 0, v, np.nan)
        return y, (lambda pred, base, idx: base * np.exp(np.asarray(pred, float) * v[idx]))
    raise ValueError(kind)


def frac_diff(s: pd.Series, dval: float, thres: float = 1e-4) -> pd.Series:
    """Fractional differencing (Lopez de Prado): reach stationarity while retaining memory.
    Uses a fixed-width causal window - only past values."""
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] * (dval - k + 1) / k
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
        if k > 500:
            break
    w = np.array(w[::-1])
    width = len(w)
    out = np.full(len(s), np.nan)
    vals = s.values.astype(float)
    for i in range(width - 1, len(s)):
        seg = vals[i - width + 1:i + 1]
        if not np.isnan(seg).any():
            out[i] = float(np.dot(w, seg))
    return pd.Series(out, index=s.index)


# =======================================================================================
# Scoring
# =======================================================================================
def score_variant(X, Y, P, folds, h, kind, vol, weight_mode="none", feats=None,
                  trunc_frac=None):
    feats = list(X.columns) if feats is None else feats
    y, inv = make_target(Y, P, h, kind, vol)
    ytrue_price = Y[f"y_price_h{h}"].values
    rows = []
    for tr, va in folds:
        mtr, mva = ~np.isnan(y[tr]), ~np.isnan(y[va])
        if mtr.sum() < 60 or mva.sum() < 10:
            continue
        tr_use = tr[mtr]
        if trunc_frac is not None:                      # 1d: recent-history-only training
            keep = max(120, int(len(tr_use) * trunc_frac))
            tr_use = tr_use[-keep:]
        Xtr = X.iloc[tr_use][feats]
        Xva = X.iloc[va][feats].iloc[mva]
        Xtr, (Xva,) = C.impute_fold(Xtr, [Xva], "native_nan")
        ytr = y[tr_use]

        sw = None
        if weight_mode == "recency":                    # 1d: exponential recency weighting
            age = np.arange(len(tr_use))[::-1]
            sw = np.exp(-age / max(60.0, len(tr_use) / 4))
        try:
            p = dict(PROBE)
            n = p.pop("n_estimators")
            m = xgb.XGBRegressor(n_estimators=n, early_stopping_rounds=50, eval_metric="rmse",
                                 random_state=C.SEED, n_jobs=-1, tree_method="hist",
                                 verbosity=0, **p)
            m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, y[va][mva])], verbose=False)
            idx = va[mva]
            pred = inv(m.predict(Xva), P.values[idx], idx)
            rows.append(C.evaluate(ytrue_price[idx], pred, P.values[idx], 0.0))
        except (xgb.core.XGBoostError, MemoryError, ValueError):
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame([_INF])


def naive_wf(Y, P, folds, h) -> float:
    rows = []
    for _, va in folds:
        yt = Y[f"y_price_h{h}"].values[va]
        b = P.values[va]
        ok = ~np.isnan(yt)
        if ok.sum() > 3:
            rows.append(float(np.sqrt(np.mean((yt[ok] - b[ok]) ** 2))))
    return float(np.mean(rows)) if rows else np.inf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if not audit_embargo():
        print("\nABORTING: fix the embargo before trusting any downstream result.")
        return 1

    d = C.load_data()
    horizons = [1, 21] if args.quick else C.HORIZONS
    resources = SIGNAL_RESOURCES + [CONTROL_RESOURCE]

    rows = []
    for res in resources:
        X, Y, P, D = C.build_dataset(d, res)
        emb = max(C.HORIZONS)
        tr, va, te = C.make_splits(len(X), emb)
        folds = C.walk_forward_folds(tr, va, emb)
        vol = np.log(P / P.shift(1)).rolling(21, min_periods=5).std()
        is_control = res == CONTROL_RESOURCE

        print(f"\n{'=' * 96}")
        print(f"{C.RESOURCES[res]['label']} ({res})"
              f"{'   [CONTROL - Stage 0 says no signal]' if is_control else '   [Stage 0: REAL SIGNAL]'}")
        print("=" * 96)

        for h in horizons:
            nb = naive_wf(Y, P, folds, h)
            base_rmse = None
            line = []
            for kind in ("price", "ret", "logret", "volnorm"):
                S = score_variant(X, Y, P, folds, h, kind, vol)
                r = float(S.RMSE.mean())
                da = float(S.DirAcc.mean())
                if kind == "logret":
                    base_rmse = r
                rows.append(dict(resource=res, horizon=h, stage="1a_target", variant=kind,
                                 rmse=r, diracc=da, naive_rmse=nb, skill=r / nb,
                                 control=is_control))
                line.append(f"{kind}={r:.1f}({r/nb:.3f})")
            print(f"  h={h:>2} naive={nb:>9.2f} | 1a targets: " + "  ".join(line))

            for wm, tf, nm in (("none", None, "full-history"),
                               ("recency", None, "recency-weighted"),
                               ("none", 0.5, "recent-50%-window")):
                S = score_variant(X, Y, P, folds, h, "logret", vol,
                                  weight_mode=wm, trunc_frac=tf)
                r = float(S.RMSE.mean())
                rows.append(dict(resource=res, horizon=h, stage="1d_regime", variant=nm,
                                 rmse=r, diracc=float(S.DirAcc.mean()), naive_rmse=nb,
                                 skill=r / nb, control=is_control))
            best_reg = min([x for x in rows if x["stage"] == "1d_regime"
                            and x["resource"] == res and x["horizon"] == h],
                           key=lambda x: x["rmse"])
            print(f"       1d regime: best = {best_reg['variant']} "
                  f"(skill {best_reg['skill']:.3f} vs logret-full {base_rmse/nb:.3f})")

    R = pd.DataFrame(rows)
    R.to_csv("stage1_results.csv", index=False)

    print("\n" + "=" * 96)
    print("STAGE 1 SUMMARY — skill ratio vs naive (< 1.0 beats naive)".center(96))
    print("=" * 96)
    for stage in ("1a_target", "1d_regime"):
        sub = R[R.stage == stage]
        print(f"\n[{stage}] mean skill ratio by variant:")
        piv = sub.pivot_table(index="variant", columns="resource", values="skill", aggfunc="mean")
        print(piv.round(4).to_string())

    print("\nBest target per resource/horizon (signal resources only):")
    sig = R[(R.stage == "1a_target") & (~R.control)]
    best = sig.loc[sig.groupby(["resource", "horizon"]).skill.idxmin()]
    print(best[["resource", "horizon", "variant", "rmse", "naive_rmse", "skill", "diracc"]]
          .round(4).to_string(index=False))

    ctrl = R[(R.stage == "1a_target") & (R.control)]
    print(f"\nCONTROL CHECK ({CONTROL_RESOURCE}): best skill = {ctrl.skill.min():.4f}")
    print("  If any variant beats naive here, treat equivalent gains on steel/cement with "
          "suspicion — Stage 0 says this series has no exploitable structure.")
    print("\nwrote stage1_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
