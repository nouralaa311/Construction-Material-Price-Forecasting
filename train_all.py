"""H21 multi-resource training pipeline.

Trains ONE dedicated model per (resource, horizon) - 5 resources x 5 horizons = 25 independent
modelling problems. Each gets its own split, its own feature ranking, its own Optuna search, and
its own saved artifact. No global multi-resource model, by design: a shared model would hide
per-resource error and block resource-specific feature engineering.

Every metric written here is what the Streamlit app displays - the app never recomputes.

Usage:
    python train_all.py                 # full run, all resources
    python train_all.py --quick         # tiny budgets, smoke test
    python train_all.py --resources steel cement
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.feature_selection import mutual_info_regression

import h21_core as C

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
XGB_STATIC = dict(random_state=C.SEED, n_jobs=-1, tree_method="hist", verbosity=0)
_INF = dict(RMSE=np.inf, MAE=np.inf, R2=-9.0, DirAcc=0.5, Coverage=0.0, N_dir=0)


# =======================================================================================
# XGBoost
# =======================================================================================
def fit_xgb(Xtr, ytr, Xva, yva, params):
    p = dict(XGB_STATIC, **params)
    n = p.pop("n_estimators", 500)
    m = xgb.XGBRegressor(n_estimators=n, early_stopping_rounds=50, eval_metric="rmse", **p)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def xgb_fold_score(params, ctx, feats, kind, strategy="native_nan", return_preds=False):
    """Walk-forward score. A fold that dies on a resource error is skipped rather than
    killing the whole Optuna study (large n_estimators x deep trees can exhaust memory)."""
    X, Y, P, h, folds = ctx["X"], ctx["Y"], ctx["P"], ctx["h"], ctx["folds"]
    ycol = f"y_{kind}_h{h}"
    scores, oof = [], []
    for tr, va in folds:
        ytr_raw, yva_raw = Y[ycol].values[tr], Y[ycol].values[va]
        mtr, mva = ~np.isnan(ytr_raw), ~np.isnan(yva_raw)
        if mtr.sum() < 50 or mva.sum() < 10:
            continue
        Xtr, Xva = X.iloc[tr][feats].iloc[mtr], X.iloc[va][feats].iloc[mva]
        Xtr, (Xva,) = C.impute_fold(Xtr, [Xva], strategy)
        base = P.values[va][mva]
        ytrue = Y[f"y_price_h{h}"].values[va][mva]
        try:
            m = fit_xgb(Xtr, ytr_raw[mtr], Xva, yva_raw[mva], params)
            pred = C.target_to_price(m.predict(Xva), base, kind)
            scores.append(C.evaluate(ytrue, pred, base, 0.0))
            if return_preds:
                oof.append((va[mva], pred, ytrue))
        except (xgb.core.XGBoostError, MemoryError):
            continue
    S = pd.DataFrame(scores) if scores else pd.DataFrame([_INF])
    return (S, oof) if return_preds else S


# =======================================================================================
# GRU / BiGRU
# =======================================================================================
class Attention(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.proj, self.v = nn.Linear(hid, hid), nn.Linear(hid, 1, bias=False)

    def forward(self, H):
        w = torch.softmax(self.v(torch.tanh(self.proj(H))).squeeze(-1), dim=1)
        return (H * w.unsqueeze(-1)).sum(1)


class GRUForecaster(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=1, dropout=0.1, arch="gru"):
        super().__init__()
        bidir = arch == "bigru"
        layers = 1 if arch == "gru" else max(2, layers) if arch == "gru2" else layers
        self.gru = nn.GRU(n_feat, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0, bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.attn = Attention(out_dim) if arch == "attn" else None
        mid = max(16, out_dim // 2)
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(dropout),
                                  nn.Linear(out_dim, mid), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(mid, 1))

    def forward(self, x):
        H, _ = self.gru(x)
        z = self.attn(H) if self.attn is not None else H[:, -1, :]
        return self.head(z).squeeze(-1)


def make_sequences(Xmat, y, seq_len, rows):
    keep = np.asarray(rows)
    keep = keep[keep >= seq_len - 1]
    keep = keep[~np.isnan(y[keep])]
    if len(keep) == 0:
        return np.empty((0, seq_len, Xmat.shape[1]), np.float32), np.empty(0, np.float32), keep
    S = np.stack([Xmat[i - seq_len + 1:i + 1] for i in keep]).astype(np.float32)
    return S, y[keep].astype(np.float32), keep


def train_gru(Str, ytr, Sva, yva, p):
    C.set_seed()
    model = GRUForecaster(Str.shape[2], int(p["hidden"]), int(p["layers"]),
                          float(p["dropout"]), p["arch"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=float(p["lr"]), weight_decay=float(p["wd"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.CFG.max_epochs)
    lossf = nn.HuberLoss() if p.get("loss", "huber") == "huber" else nn.MSELoss()
    Xtr_t, ytr_t = torch.tensor(Str), torch.tensor(ytr)
    Xva_t, yva_t = torch.tensor(Sva).to(DEVICE), torch.tensor(yva).to(DEVICE)
    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    bs = int(p["batch"])
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(ds) > bs)
    best, best_state, bad = np.inf, None, 0
    for _ in range(C.CFG.max_epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(nn.functional.mse_loss(model(Xva_t), yva_t))
        sched.step()
        if vl < best - 1e-12:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= C.CFG.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def gru_predict(model, S, bs=512):
    if len(S) == 0:
        return np.empty(0)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(S), bs):
            out.append(model(torch.tensor(S[i:i + bs]).to(DEVICE)).cpu().numpy())
    return np.concatenate(out)


def gru_fold_score(p, ctx, feats, kind, return_preds=False):
    X, Y, P, h, folds = ctx["X"], ctx["Y"], ctx["P"], ctx["h"], ctx["folds"]
    y = Y[f"y_{kind}_h{h}"].values
    L = int(p["seq_len"])
    rows, oof = [], []
    for tr, va in folds:
        Xtr_df, (Xfull,) = C.impute_fold(X.iloc[tr][feats], [X[feats]], "ffill_median")
        fsc = C.SCALERS[p.get("scaler", "standard")]().fit(Xtr_df.values)
        Xm = np.clip(np.nan_to_num(fsc.transform(Xfull.values).astype(np.float32)), -10, 10)
        ytr_valid = y[tr][~np.isnan(y[tr])]
        if len(ytr_valid) < 50:
            continue
        ysc = C.SCALERS["standard"]().fit(ytr_valid.reshape(-1, 1))
        yz = np.full_like(y, np.nan, dtype=float)
        ok = ~np.isnan(y)
        yz[ok] = ysc.transform(y[ok].reshape(-1, 1)).ravel()
        Str, ytr_s, _ = make_sequences(Xm, yz, L, tr)
        Sva, yva_s, kva = make_sequences(Xm, yz, L, va)
        if len(Str) < 60 or len(Sva) < 10:
            continue
        model = train_gru(Str, ytr_s, Sva, yva_s, p)
        pred = ysc.inverse_transform(gru_predict(model, Sva).reshape(-1, 1)).ravel()
        base = P.values[kva]
        ytrue = Y[f"y_price_h{h}"].values[kva]
        pp = C.target_to_price(pred, base, kind)
        rows.append(C.evaluate(ytrue, pp, base, 0.0))
        if return_preds:
            oof.append((kva, pp, ytrue))
    S = pd.DataFrame(rows) if rows else pd.DataFrame([_INF])
    return (S, oof) if return_preds else S


# =======================================================================================
# Per-(resource, horizon) training
# =======================================================================================
def select_features(X, Y, tr, h, kind, k):
    y = Y[f"y_{kind}_h{h}"].values[tr]
    m = ~np.isnan(y)
    Xs, ys = X.iloc[tr].iloc[m], y[m]
    Xi = Xs.fillna(Xs.median()).fillna(0.0)
    C.set_seed()
    rank = pd.DataFrame(index=X.columns)
    rank["mi"] = mutual_info_regression(Xi, ys, random_state=C.SEED)
    rank["corr"] = np.abs([np.corrcoef(Xi[c], ys)[0, 1] if Xi[c].std() > 0 else 0.0 for c in X.columns])
    g = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.7, reg_lambda=2.0, random_state=C.SEED,
                         n_jobs=-1, tree_method="hist")
    g.fit(Xs, ys)
    rank["gain"] = g.feature_importances_
    rank["consensus"] = rank.rank(pct=True).mean(axis=1)
    ranked = rank.sort_values("consensus", ascending=False)
    return list(ranked.index[:k]), ranked


def train_one(d, resource, h, quick=False):
    t0 = time.time()
    X, Y, P, D = C.build_dataset(d, resource)
    emb = max(C.HORIZONS)
    tr, va, te = C.make_splits(len(X), emb)
    folds = C.walk_forward_folds(tr, va, emb)
    ctx = dict(X=X, Y=Y, P=P, h=h, folds=folds)

    # ---- target formulation chosen on walk-forward validation only -------------------
    feats_probe, _ = select_features(X, Y, tr, h, "logret", C.CFG.top_k_features)
    kind_scores = {}
    for kind in ("price", "ret", "logret"):
        S = xgb_fold_score(dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                                subsample=0.85, colsample_bytree=0.7, min_child_weight=4,
                                reg_lambda=2.0), ctx, feats_probe, kind)
        kind_scores[kind] = float(S.RMSE.mean())
    kind = min(kind_scores, key=kind_scores.get)

    feats, ranked = select_features(X, Y, tr, h, kind, C.CFG.top_k_features)

    # ---- imputation strategy on validation -------------------------------------------
    strat_scores = {}
    for s in ("native_nan", "ffill_median", "median"):
        S = xgb_fold_score(dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                                subsample=0.85, colsample_bytree=0.7, min_child_weight=4,
                                reg_lambda=2.0), ctx, feats, kind, s)
        strat_scores[s] = float(S.RMSE.mean())
    strategy = min(strat_scores, key=strat_scores.get)

    # ---- Optuna: XGBoost --------------------------------------------------------------
    def xgb_obj(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators", 200, 900, step=100),
                 max_depth=trial.suggest_int("max_depth", 2, 7),
                 learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.25, log=True),
                 min_child_weight=trial.suggest_int("min_child_weight", 1, 25),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                 gamma=trial.suggest_float("gamma", 0.0, 5.0),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True))
        v = float(xgb_fold_score(p, ctx, feats, kind, strategy).RMSE.mean())
        return v if np.isfinite(v) else 1e12

    C.set_seed()
    s_xgb = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED, n_startup_trials=6))
    s_xgb.optimize(xgb_obj, n_trials=3 if quick else C.CFG.n_trials_xgb, show_progress_bar=False)
    best_xgb_params = s_xgb.best_params
    S_xgb = xgb_fold_score(best_xgb_params, ctx, feats, kind, strategy)

    # ---- Optuna: GRU ------------------------------------------------------------------
    gru_feats = feats[:40]

    def gru_obj(trial):
        p = dict(arch=trial.suggest_categorical("arch", ["gru", "gru2", "attn", "bigru"]),
                 seq_len=trial.suggest_categorical("seq_len", [5, 7, 10, 14, 21, 30]),
                 hidden=trial.suggest_categorical("hidden", [32, 64, 96, 128]),
                 layers=trial.suggest_int("layers", 1, 2),
                 dropout=trial.suggest_float("dropout", 0.05, 0.45),
                 lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                 wd=trial.suggest_float("wd", 1e-6, 1e-2, log=True),
                 batch=trial.suggest_categorical("batch", [16, 32, 64]),
                 loss=trial.suggest_categorical("loss", ["huber", "mse"]),
                 scaler="standard")
        v = float(gru_fold_score(p, ctx, gru_feats, kind).RMSE.mean())
        return v if np.isfinite(v) else 1e12

    C.set_seed()
    s_gru = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED, n_startup_trials=4))
    s_gru.optimize(gru_obj, n_trials=2 if quick else C.CFG.n_trials_gru, show_progress_bar=False)
    best_gru_params = dict(s_gru.best_params, scaler="standard")
    S_gru = gru_fold_score(best_gru_params, ctx, gru_feats, kind)

    # ---- pick winner on VALIDATION, then freeze ---------------------------------------
    val_xgb, val_gru = float(S_xgb.RMSE.mean()), float(S_gru.RMSE.mean())
    winner = "xgboost" if val_xgb <= val_gru else "gru"

    # ---- dead zone on validation OOF --------------------------------------------------
    if winner == "xgboost":
        _, oof = xgb_fold_score(best_xgb_params, ctx, feats, kind, strategy, return_preds=True)
    else:
        _, oof = gru_fold_score(best_gru_params, ctx, gru_feats, kind, return_preds=True)
    if oof:
        oi = np.concatenate([o[0] for o in oof])
        op = np.concatenate([o[1] for o in oof])
        ot = np.concatenate([o[2] for o in oof])
        ob = P.values[oi]
        dz_rows = []
        for dz in C.CFG.dead_zones:
            s = C.directional_stats(ot, op, ob, dz)
            dz_rows.append(s)
        dzt = pd.DataFrame(dz_rows)
        ok = dzt[(dzt.coverage >= C.CFG.coverage_floor) & dzt.dir_acc.notna()]
        best_dz = float((ok if len(ok) else dzt).sort_values("dir_acc", ascending=False).iloc[0].dead_zone)
    else:
        best_dz = 0.0

    # ---- final refit on TRAIN+VAL, evaluate once on TEST -------------------------------
    DEV = np.concatenate([tr, va])
    ycol = f"y_{kind}_h{h}"
    yk = Y[ycol].values
    base_te = P.values[te]
    ytrue_te = Y[f"y_price_h{h}"].values[te]
    ok_te = ~np.isnan(ytrue_te)
    inner = int(len(DEV) * 0.85)

    artifact = dict(resource=resource, horizon=h, target_kind=kind, winner=winner,
                    strategy=strategy, dead_zone=best_dz, features=feats, gru_features=gru_feats)

    if winner == "xgboost":
        Xdev_i, (Xte_i,) = C.impute_fold(X.iloc[DEV][feats], [X.iloc[te][feats]], strategy)
        ya, yb = yk[DEV][:inner], yk[DEV][inner:]
        ma, mb = ~np.isnan(ya), ~np.isnan(yb)
        model = fit_xgb(Xdev_i.iloc[:inner][ma], ya[ma], Xdev_i.iloc[inner:][mb], yb[mb], best_xgb_params)
        pred_te = C.target_to_price(model.predict(Xte_i), base_te, kind)
        yt_eval, pred_eval, base_eval = ytrue_te[ok_te], pred_te[ok_te], base_te[ok_te]
        artifact["model"] = model
        artifact["params"] = best_xgb_params
    else:
        Xd_i, (Xfull,) = C.impute_fold(X.iloc[DEV][gru_feats], [X[gru_feats]], "ffill_median")
        fsc = C.SCALERS[best_gru_params["scaler"]]().fit(Xd_i.values)
        Xm = np.clip(np.nan_to_num(fsc.transform(Xfull.values).astype(np.float32)), -10, 10)
        ysc = C.SCALERS["standard"]().fit(yk[DEV][~np.isnan(yk[DEV])].reshape(-1, 1))
        yz = np.full_like(yk, np.nan, float)
        okk = ~np.isnan(yk)
        yz[okk] = ysc.transform(yk[okk].reshape(-1, 1)).ravel()
        L = int(best_gru_params["seq_len"])
        Str, ytr_s, _ = make_sequences(Xm, yz, L, DEV[:inner])
        Sva, yva_s, _ = make_sequences(Xm, yz, L, DEV[inner:])
        model = train_gru(Str, ytr_s, Sva, yva_s, best_gru_params)
        Ste, _, kte = make_sequences(Xm, np.zeros_like(yz), L, te)
        raw = ysc.inverse_transform(gru_predict(model, Ste).reshape(-1, 1)).ravel()
        pred_full = C.target_to_price(raw, P.values[kte], kind)
        yt_all = Y[f"y_price_h{h}"].values[kte]
        m2 = ~np.isnan(yt_all)
        yt_eval, pred_eval, base_eval = yt_all[m2], pred_full[m2], P.values[kte][m2]
        artifact["model"] = model.state_dict()
        artifact["params"] = best_gru_params
        artifact["feature_scaler"] = fsc
        artifact["target_scaler"] = ysc
        artifact["n_features_gru"] = Xm.shape[1]

    test_row = C.evaluate(yt_eval, pred_eval, base_eval, best_dz, winner, h)
    test_full = C.evaluate(yt_eval, pred_eval, base_eval, 0.0, winner, h)
    naive_row = C.evaluate(yt_eval, base_eval, base_eval, best_dz, "naive", h)
    dm_stat, dm_p = C.diebold_mariano(yt_eval, pred_eval, base_eval)

    # best-of-any-threshold (reported, never selected on)
    best_any = max((C.directional_stats(yt_eval, pred_eval, base_eval, dz)
                    for dz in C.CFG.dead_zones),
                   key=lambda s: (-1 if s["dir_acc"] is None or np.isnan(s["dir_acc"]) else s["dir_acc"]))

    artifact["metrics"] = dict(
        val_rmse_xgb=val_xgb, val_rmse_gru=val_gru,
        val_rmse=min(val_xgb, val_gru),
        val_r2=float(S_xgb.R2.mean() if winner == "xgboost" else S_gru.R2.mean()),
        val_diracc=float(S_xgb.DirAcc.mean() if winner == "xgboost" else S_gru.DirAcc.mean()),
        test_rmse=test_row["RMSE"], test_mae=test_row["MAE"], test_r2=test_row["R2"],
        test_mape=test_row["MAPE%"],
        test_diracc=test_row["DirAcc"], test_coverage=test_row["Coverage"],
        test_n_dir=test_row["N_dir"], test_p_value=test_row["p_value"],
        test_precision=test_row["Precision"], test_recall=test_row["Recall"], test_f1=test_row["F1"],
        test_cm=test_row["cm"],
        test_diracc_full=test_full["DirAcc"], test_n_dir_full=test_full["N_dir"],
        best_any_diracc=best_any["dir_acc"], best_any_dz=best_any["dead_zone"],
        best_any_coverage=best_any["coverage"],
        naive_rmse=naive_row["RMSE"], dm_stat=dm_stat, dm_p=dm_p,
        beats_naive=bool(test_row["RMSE"] < naive_row["RMSE"]),
        beats_naive_significant=bool(dm_p < 0.05 and dm_stat < 0),
    )
    artifact["meta"] = dict(
        resource=resource, label=C.RESOURCES[resource]["label"], unit=C.RESOURCES[resource]["unit"],
        horizon=h, approx_days=C.HORIZON_DAYS[h], target_kind=kind, winner=winner,
        strategy=strategy, dead_zone=best_dz,
        n_samples=len(X), n_features=len(feats),
        train_start=str(D.iloc[tr[0]].date()), train_end=str(D.iloc[tr[-1]].date()),
        test_start=str(D.iloc[te[0]].date()), test_end=str(D.iloc[te[-1]].date()),
        n_train=len(tr), n_val=len(va), n_test=len(te),
        elapsed_s=round(time.time() - t0, 1),
        top_features=list(ranked.index[:15]),
    )
    return artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resources", nargs="*", default=list(C.RESOURCES))
    ap.add_argument("--horizons", nargs="*", type=int, default=C.HORIZONS)
    args = ap.parse_args()

    os.makedirs(C.MODEL_DIR, exist_ok=True)
    d = C.load_data()
    C.assert_no_forbidden_patterns()
    print(f"data: {d.shape} | device={DEVICE} | quick={args.quick}")

    rows = []
    for r in args.resources:
        C.assert_causal(d, r)
        print(f"\n[{r}] causality assertion PASSED")
        for h in args.horizons:
            art = train_one(d, r, h, quick=args.quick)
            path = os.path.join(C.MODEL_DIR, f"{r}_h{h}.joblib")
            joblib.dump(art, path)
            with open(os.path.join(C.MODEL_DIR, f"{r}_h{h}.meta.json"), "w") as fh:
                json.dump({"meta": art["meta"], "metrics": art["metrics"],
                           "params": art["params"], "features": art["features"]},
                          fh, indent=2, default=str)
            m = art["metrics"]
            rows.append(dict(resource=r, horizon=h, approx_days=C.HORIZON_DAYS[h],
                             winner=art["winner"], target=art["target_kind"],
                             **{k: m[k] for k in ("test_rmse", "test_mae", "test_r2",
                                                  "test_diracc", "test_coverage", "test_n_dir",
                                                  "test_p_value", "naive_rmse", "dm_p",
                                                  "beats_naive", "best_any_diracc")}))
            print(f"  h={h:>2} {art['winner']:<8} RMSE={m['test_rmse']:>9.2f} "
                  f"R2={m['test_r2']:>7.3f} DirAcc={m['test_diracc'] if m['test_diracc'] is not None else float('nan'):>6.3f} "
                  f"cov={m['test_coverage']:.0%} beats_naive={m['beats_naive']} "
                  f"({art['meta']['elapsed_s']}s)")

    LB = pd.DataFrame(rows)
    LB.to_csv("leaderboard.csv", index=False)
    print("\n" + "=" * 100)
    print("CONSOLIDATED LEADERBOARD".center(100))
    print("=" * 100)
    print(LB.to_string(index=False))
    print(f"\nresource/horizon combos that BEAT the naive baseline: "
          f"{int(LB.beats_naive.sum())} / {len(LB)}")
    print("wrote leaderboard.csv")


if __name__ == "__main__":
    main()
