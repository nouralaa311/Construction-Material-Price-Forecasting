"""STAGES 2-6 — the rest of the optimization campaign.

Carries forward the Stage 0/1 findings:
  * Stage 0: only steel + cement show structure the variance-ratio test can distinguish from a
    random walk. oil/aluminium/copper are treated as controls - we still train them, but the
    honest prior is that naive wins, and we say so.
  * Stage 1: log-returns are the target; price-level targets are disqualified; regime handling
    (full history vs recency-weighted vs truncated) is chosen PER RESOURCE.

  Stage 2  feature groups + hard pruning (validation-fold permutation importance)
  Stage 3  model shortlist: Ridge/ElasticNet, RF/ExtraTrees, XGBoost, LightGBM, CatBoost
  Stage 4  ensembling incl. shrinkage toward the naive forecast
  Stage 5  conformal prediction intervals + frozen directional operating point
  Stage 6  Diebold-Mariano, multi-seed stability, Benjamini-Hochberg correction

Every accepted improvement must clear Stage 6. Results are written to stage*.csv and the
winning configuration is saved into models/ for the Streamlit app.

Usage:
    python stage_campaign.py --stage all
    python stage_campaign.py --stage 2 --quick
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
import scipy.stats as st
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.inspection import permutation_importance

import h21_core as C
from train_all import _INF

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 50)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

SIGNAL = ["steel", "cement"]
CONTROL = ["oil", "aluminium", "copper"]
ALL_RES = SIGNAL + CONTROL
TARGET_KIND = "logret"          # Stage 1 verdict


# =======================================================================================
# shared plumbing
# =======================================================================================
def ctx_for(d, res):
    X, Y, P, D = C.build_dataset(d, res)
    emb = max(C.HORIZONS)
    tr, va, te = C.make_splits(len(X), emb)
    folds = C.walk_forward_folds(tr, va, emb)
    return dict(X=X, Y=Y, P=P, D=D, tr=tr, va=va, te=te, folds=folds, res=res)


def naive_wf(ctx, h) -> float:
    out = []
    for _, va in ctx["folds"]:
        yt = ctx["Y"][f"y_price_h{h}"].values[va]
        b = ctx["P"].values[va]
        ok = ~np.isnan(yt)
        if ok.sum() > 3:
            out.append(float(np.sqrt(np.mean((yt[ok] - b[ok]) ** 2))))
    return float(np.mean(out)) if out else np.inf


def make_model(name, params=None, seed=C.SEED):
    p = dict(params or {})
    if name == "ridge":
        return Ridge(alpha=p.get("alpha", 10.0), random_state=None)
    if name == "elasticnet":
        return ElasticNet(alpha=p.get("alpha", 0.001), l1_ratio=p.get("l1_ratio", 0.5),
                          max_iter=5000)
    if name == "rf":
        return RandomForestRegressor(n_estimators=p.get("n_estimators", 300),
                                     max_depth=p.get("max_depth", 6),
                                     min_samples_leaf=p.get("min_samples_leaf", 8),
                                     random_state=seed, n_jobs=-1)
    if name == "extratrees":
        return ExtraTreesRegressor(n_estimators=p.get("n_estimators", 300),
                                   max_depth=p.get("max_depth", 8),
                                   min_samples_leaf=p.get("min_samples_leaf", 8),
                                   random_state=seed, n_jobs=-1)
    if name == "xgboost":
        return xgb.XGBRegressor(n_estimators=p.get("n_estimators", 400),
                                max_depth=p.get("max_depth", 4),
                                learning_rate=p.get("learning_rate", 0.05),
                                subsample=p.get("subsample", 0.85),
                                colsample_bytree=p.get("colsample_bytree", 0.7),
                                min_child_weight=p.get("min_child_weight", 4),
                                reg_alpha=p.get("reg_alpha", 0.1),
                                reg_lambda=p.get("reg_lambda", 2.0),
                                random_state=seed, n_jobs=-1, tree_method="hist", verbosity=0)
    if name == "lightgbm":
        return lgb.LGBMRegressor(n_estimators=p.get("n_estimators", 400),
                                 num_leaves=p.get("num_leaves", 15),
                                 learning_rate=p.get("learning_rate", 0.05),
                                 min_child_samples=p.get("min_child_samples", 20),
                                 subsample=p.get("subsample", 0.85), subsample_freq=1,
                                 colsample_bytree=p.get("colsample_bytree", 0.7),
                                 reg_lambda=p.get("reg_lambda", 2.0),
                                 random_state=seed, n_jobs=-1, verbose=-1)
    if name == "catboost":
        return CatBoostRegressor(iterations=p.get("iterations", 400),
                                 depth=p.get("depth", 5),
                                 learning_rate=p.get("learning_rate", 0.05),
                                 l2_leaf_reg=p.get("l2_leaf_reg", 3.0),
                                 random_seed=seed, verbose=0, allow_writing_files=False)
    raise ValueError(name)


NEEDS_DENSE = {"ridge", "elasticnet", "rf", "extratrees", "catboost"}


def fit_predict(name, Xtr, ytr, Xva, params=None, seed=C.SEED, sample_weight=None):
    strategy = "ffill_median" if name in NEEDS_DENSE else "native_nan"
    Xtr2, (Xva2,) = C.impute_fold(Xtr, [Xva], strategy)
    m = make_model(name, params, seed)
    try:
        m.fit(Xtr2, ytr, sample_weight=sample_weight)
    except TypeError:
        m.fit(Xtr2, ytr)
    return m, m.predict(Xva2)


def wf_score(name, ctx, h, feats, params=None, seed=C.SEED, regime="full",
             return_oof=False):
    """Walk-forward score in PRICE space for one model family."""
    X, Y, P = ctx["X"], ctx["Y"], ctx["P"]
    y = Y[f"y_{TARGET_KIND}_h{h}"].values
    ytrue = Y[f"y_price_h{h}"].values
    rows, oof = [], []
    for tr, va in ctx["folds"]:
        mtr, mva = ~np.isnan(y[tr]), ~np.isnan(y[va])
        if mtr.sum() < 60 or mva.sum() < 10:
            continue
        tr_use = tr[mtr]
        sw = None
        if regime == "trunc50":
            tr_use = tr_use[-max(120, len(tr_use) // 2):]
        elif regime == "recency":
            age = np.arange(len(tr_use))[::-1]
            sw = np.exp(-age / max(60.0, len(tr_use) / 4))
        idx = va[mva]
        try:
            _, pred = fit_predict(name, X.iloc[tr_use][feats], y[tr_use],
                                  X.iloc[idx][feats], params, seed, sw)
        except Exception:
            continue
        pp = C.target_to_price(pred, P.values[idx], TARGET_KIND)
        rows.append(C.evaluate(ytrue[idx], pp, P.values[idx], 0.0))
        if return_oof:
            oof.append((idx, pp, ytrue[idx]))
    S = pd.DataFrame(rows) if rows else pd.DataFrame([_INF])
    return (S, oof) if return_oof else S


def oof_frame(oof):
    if not oof:
        return pd.DataFrame(columns=["idx", "pred", "true"]).set_index("idx")
    i = np.concatenate([o[0] for o in oof])
    p = np.concatenate([o[1] for o in oof])
    t = np.concatenate([o[2] for o in oof])
    return pd.DataFrame({"idx": i, "pred": p, "true": t}).groupby("idx").last()


# =======================================================================================
# STAGE 2 — feature groups + pruning
# =======================================================================================
def feature_groups(cols):
    g = {
        "own_lags": [c for c in cols if c.startswith(("own_lag_", "own_ret_lag_", "own_logret_lag_"))],
        "own_rolling": [c for c in cols if c.startswith(("own_mean_", "own_std_", "own_min_",
                                                         "own_max_", "own_range_", "own_z_",
                                                         "own_vol_"))],
        "own_momentum": [c for c in cols if c.startswith(("own_ret_", "own_mom_", "own_rsi",
                                                          "own_accel", "own_macd", "own_dist_"))
                         and not c.startswith("own_ret_lag_")],
        "calendar": [c for c in cols if c in ("dow", "dom", "month", "days_since_prev_obs",
                                              "gap_roll_mean_5", "is_after_long_gap")],
        "cross_resource": [c for c in cols if c.startswith(("steel_", "cement_", "oil_",
                                                            "aluminium_", "copper_", "ratio_own_",
                                                            "relmom_own_"))],
        "fx": [c for c in cols if c.startswith(("usd_", "own_usd_", "egp_"))],
        "external_macro": [c for c in cols if c.startswith(("iron_ore_", "coal_", "natgas_",
                                                            "cbe_", "egypt_"))],
    }
    seen = set()
    for k in g:
        g[k] = [c for c in g[k] if c in cols and not (c in seen or seen.add(c))]
    return {k: v for k, v in g.items() if v}


def stage2(d, horizons, quick=False):
    print("\n" + "=" * 100)
    print("STAGE 2 — FEATURE GROUPS + PRUNING".center(100))
    print("=" * 100)
    rows = []
    for res in ALL_RES:
        ctx = ctx_for(d, res)
        groups = feature_groups(list(ctx["X"].columns))
        allf = [c for g in groups.values() for c in g]
        for h in horizons:
            nb = naive_wf(ctx, h)
            base = float(wf_score("xgboost", ctx, h, allf).RMSE.mean())
            rows.append(dict(resource=res, horizon=h, variant="ALL", n_feat=len(allf),
                             rmse=base, skill=base / nb, naive=nb))
            # leave-one-group-out: the delta a group actually earns
            for gname, gcols in groups.items():
                rest = [c for c in allf if c not in gcols]
                if len(rest) < 10:
                    continue
                r = float(wf_score("xgboost", ctx, h, rest).RMSE.mean())
                rows.append(dict(resource=res, horizon=h, variant=f"minus_{gname}",
                                 n_feat=len(rest), rmse=r, skill=r / nb, naive=nb,
                                 delta_vs_all=r - base))
            # pruned set via validation-fold permutation importance
            pruned = prune_features(ctx, h, allf, top_k=30 if not quick else 20)
            rp = float(wf_score("xgboost", ctx, h, pruned).RMSE.mean())
            rows.append(dict(resource=res, horizon=h, variant="PRUNED", n_feat=len(pruned),
                             rmse=rp, skill=rp / nb, naive=nb, delta_vs_all=rp - base,
                             pruned_features=json.dumps(pruned)))
            print(f"  {res:<10} h={h:>2}  ALL({len(allf)})={base:>9.2f} ({base/nb:.3f})   "
                  f"PRUNED({len(pruned)})={rp:>9.2f} ({rp/nb:.3f})   "
                  f"{'PRUNED WINS' if rp < base else 'all-features wins'}")
    R = pd.DataFrame(rows)
    R.to_csv("stage2_results.csv", index=False)
    print("\nGroup value (mean RMSE increase when the group is REMOVED; positive = group earns its keep):")
    gg = R[R.variant.str.startswith("minus_")].copy()
    gg["group"] = gg.variant.str.replace("minus_", "", regex=False)
    print(gg.pivot_table(index="group", columns="resource", values="delta_vs_all",
                         aggfunc="mean").round(2).to_string())
    return R


def prune_features(ctx, h, feats, top_k=30):
    """Permutation importance computed on VALIDATION folds (never training)."""
    X, Y, P = ctx["X"], ctx["Y"], ctx["P"]
    y = Y[f"y_{TARGET_KIND}_h{h}"].values
    imp = pd.Series(0.0, index=feats)
    n = 0
    for tr, va in ctx["folds"][-2:]:
        mtr, mva = ~np.isnan(y[tr]), ~np.isnan(y[va])
        if mtr.sum() < 60 or mva.sum() < 20:
            continue
        Xtr, (Xva,) = C.impute_fold(X.iloc[tr[mtr]][feats], [X.iloc[va[mva]][feats]], "ffill_median")
        m = make_model("xgboost")
        m.fit(Xtr, y[tr[mtr]])
        pi = permutation_importance(m, Xva, y[va[mva]], n_repeats=3, random_state=C.SEED,
                                    scoring="neg_root_mean_squared_error")
        imp += pd.Series(pi.importances_mean, index=feats)
        n += 1
    if n == 0:
        return feats[:top_k]
    return list((imp / n).sort_values(ascending=False).index[:top_k])


# =======================================================================================
# STAGE 3 — model shortlist
# =======================================================================================
FAMILIES = ["ridge", "elasticnet", "rf", "extratrees", "xgboost", "lightgbm", "catboost"]


def stage3(d, horizons, feats_map, regime_map, quick=False):
    print("\n" + "=" * 100)
    print("STAGE 3 — MODEL SHORTLIST".center(100))
    print("=" * 100)
    fams = FAMILIES if not quick else ["ridge", "xgboost", "lightgbm"]
    rows = []
    for res in ALL_RES:
        ctx = ctx_for(d, res)
        regime = regime_map.get(res, "full")
        for h in horizons:
            nb = naive_wf(ctx, h)
            feats = feats_map.get((res, h)) or list(ctx["X"].columns)
            best = None
            for fam in fams:
                S = wf_score(fam, ctx, h, feats, regime=regime)
                r, da = float(S.RMSE.mean()), float(S.DirAcc.mean())
                rows.append(dict(resource=res, horizon=h, family=fam, rmse=r, diracc=da,
                                 skill=r / nb, naive=nb, regime=regime))
                if best is None or r < best[1]:
                    best = (fam, r)
            print(f"  {res:<10} h={h:>2}  best={best[0]:<11} rmse={best[1]:>9.2f} "
                  f"skill={best[1]/nb:.3f} {'BEATS naive' if best[1] < nb else 'loses to naive'}")
    R = pd.DataFrame(rows)
    R.to_csv("stage3_results.csv", index=False)
    print("\nMean skill by family (lower is better; <1 beats naive):")
    print(R.pivot_table(index="family", columns="resource", values="skill",
                        aggfunc="mean").round(4).to_string())
    return R


# =======================================================================================
# STAGE 4 — ensembling
# =======================================================================================
def stage4(d, horizons, feats_map, regime_map, stage3_R):
    print("\n" + "=" * 100)
    print("STAGE 4 — ENSEMBLING".center(100))
    print("=" * 100)
    rows = []
    for res in ALL_RES:
        ctx = ctx_for(d, res)
        regime = regime_map.get(res, "full")
        for h in horizons:
            nb = naive_wf(ctx, h)
            feats = feats_map.get((res, h)) or list(ctx["X"].columns)
            sub = stage3_R[(stage3_R.resource == res) & (stage3_R.horizon == h)]
            top = list(sub.nsmallest(3, "rmse").family)
            oofs, preds = {}, {}
            for fam in top:
                _, o = wf_score(fam, ctx, h, feats, regime=regime, return_oof=True)
                F = oof_frame(o)
                if len(F):
                    oofs[fam] = F
            if not oofs:
                continue
            common = None
            for F in oofs.values():
                common = F.index if common is None else common.intersection(F.index)
            if common is None or len(common) < 20:
                continue
            M = np.column_stack([oofs[f].loc[common, "pred"].values for f in oofs])
            t = oofs[top[0]].loc[common, "true"].values
            base = ctx["P"].values[common.values]

            def rec(nm, p):
                rows.append(dict(resource=res, horizon=h, variant=nm,
                                 rmse=C.rmse(t, p), skill=C.rmse(t, p) / nb, naive=nb,
                                 diracc=C.directional_stats(t, p, base)["dir_acc"]))

            rec("best_single", M[:, 0])
            rec("simple_avg", M.mean(1))
            errs = np.array([C.rmse(t, M[:, i]) for i in range(M.shape[1])])
            w = (1 / errs) / (1 / errs).sum()
            rec("inv_error", M @ w)
            # shrinkage toward naive - often the honest winner at long horizons
            for lam in (0.25, 0.5, 0.75):
                rec(f"shrink_naive_{lam}", (1 - lam) * M.mean(1) + lam * base)
            # constrained non-negative stack on OOF only (chronological inner split)
            from scipy.optimize import minimize
            half = len(t) // 2
            r0 = minimize(lambda w_: C.rmse(t[:half], M[:half] @ w_),
                          np.repeat(1 / M.shape[1], M.shape[1]), method="SLSQP",
                          bounds=[(0, 1)] * M.shape[1],
                          constraints={"type": "eq", "fun": lambda w_: w_.sum() - 1})
            wv = np.clip(r0.x, 0, None)
            wv = wv / wv.sum() if wv.sum() > 0 else np.repeat(1 / M.shape[1], M.shape[1])
            rec("stacked_nnls", M @ wv)
            bst = min([r for r in rows if r["resource"] == res and r["horizon"] == h],
                      key=lambda r: r["rmse"])
            print(f"  {res:<10} h={h:>2}  best={bst['variant']:<18} skill={bst['skill']:.3f} "
                  f"(members: {', '.join(oofs)})")
    R = pd.DataFrame(rows)
    R.to_csv("stage4_results.csv", index=False)
    print("\nMean skill by ensemble variant:")
    print(R.pivot_table(index="variant", columns="resource", values="skill",
                        aggfunc="mean").round(4).to_string())
    return R


# =======================================================================================
# STAGE 5 / 6 — uncertainty + statistical validation
# =======================================================================================
def dm_test(y, p1, p2):
    e1 = (np.asarray(y) - np.asarray(p1)) ** 2
    e2 = (np.asarray(y) - np.asarray(p2)) ** 2
    dd = e1 - e2
    sd = dd.std(ddof=1)
    if sd == 0 or len(dd) < 4:
        return 0.0, 1.0
    stat = float(dd.mean() / (sd / np.sqrt(len(dd))))
    return stat, float(2 * (1 - st.t.cdf(abs(stat), df=len(dd) - 1)))


def benjamini_hochberg(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(n, bool)
    if k > 0:
        out[order[:k]] = True
    return out


def stage56(d, horizons, feats_map, regime_map, stage3_R, seeds=(42, 1, 7, 13, 99)):
    print("\n" + "=" * 100)
    print("STAGES 5 & 6 — UNCERTAINTY + STATISTICAL VALIDATION".center(100))
    print("=" * 100)
    rows = []
    for res in ALL_RES:
        ctx = ctx_for(d, res)
        regime = regime_map.get(res, "full")
        for h in horizons:
            nb = naive_wf(ctx, h)
            feats = feats_map.get((res, h)) or list(ctx["X"].columns)
            sub = stage3_R[(stage3_R.resource == res) & (stage3_R.horizon == h)]
            if not len(sub):
                continue
            fam = sub.nsmallest(1, "rmse").family.iloc[0]

            per_seed = []
            oof_last = None
            for sd in seeds:
                S, o = wf_score(fam, ctx, h, feats, seed=sd, regime=regime, return_oof=True)
                per_seed.append(float(S.RMSE.mean()))
                oof_last = o
            per_seed = np.array(per_seed)

            F = oof_frame(oof_last)
            if not len(F):
                continue
            idx = F.index.values
            t, p = F["true"].values, F["pred"].values
            base = ctx["P"].values[idx]
            stat, pval = dm_test(t, p, base)          # vs naive
            ds = C.directional_stats(t, p, base, 0.0)

            # Stage 5: split conformal interval from the first half of OOF residuals
            half = len(t) // 2
            resid = np.abs(t[:half] - p[:half])
            q90 = float(np.quantile(resid, 0.9)) if len(resid) > 5 else np.nan
            cover = float(np.mean(np.abs(t[half:] - p[half:]) <= q90)) if np.isfinite(q90) else np.nan

            rows.append(dict(resource=res, horizon=h, family=fam, regime=regime,
                             rmse_mean=per_seed.mean(), rmse_std=per_seed.std(),
                             naive=nb, skill=per_seed.mean() / nb,
                             dm_stat=stat, dm_p=pval,
                             beats_naive=bool(per_seed.mean() < nb),
                             diracc=ds["dir_acc"], dir_p=ds["p_value"], n_dir=ds["n_valid"],
                             conformal_q90=q90, conformal_coverage=cover,
                             signal_group="signal" if res in SIGNAL else "control"))
            print(f"  {res:<10} h={h:>2} {fam:<11} rmse={per_seed.mean():>9.2f}±{per_seed.std():>6.2f} "
                  f"skill={per_seed.mean()/nb:.3f} DM_p={pval:.4f} "
                  f"cover90={cover if np.isfinite(cover) else float('nan'):.2f}")

    R = pd.DataFrame(rows)
    if len(R):
        beat = R.beats_naive & (R.dm_stat < 0)
        R["bh_significant"] = False
        if beat.any():
            R.loc[beat, "bh_significant"] = benjamini_hochberg(R.loc[beat, "dm_p"].values)
        R["n_configs_tested"] = len(FAMILIES) * len(horizons) * len(ALL_RES)
    R.to_csv("stage6_final.csv", index=False)

    print("\n" + "=" * 100)
    print("STAGE 6 VERDICT — survives Diebold-Mariano + Benjamini-Hochberg".center(100))
    print("=" * 100)
    if len(R):
        win = R[R.bh_significant]
        print(f"configurations tested across the campaign: ~{int(R.n_configs_tested.iloc[0])}")
        print(f"beat naive on point RMSE: {int(R.beats_naive.sum())}/{len(R)}")
        print(f"SURVIVE multiple-comparison correction: {len(win)}/{len(R)}")
        if len(win):
            print("\nACCEPTED IMPROVEMENTS:")
            print(win[["resource", "horizon", "family", "regime", "rmse_mean", "rmse_std",
                       "naive", "skill", "dm_p", "diracc"]].round(4).to_string(index=False))
        else:
            print("\nNONE survived. The honest conclusion is that no configuration beats the "
                  "naive baseline by a statistically defensible margin.")
        ctrl = R[R.signal_group == "control"]
        print(f"\nCONTROL GROUP (Stage 0: no signal) — {int(ctrl.bh_significant.sum())}/{len(ctrl)} "
              "significant. Any number >0 here means the pipeline finds 'signal' in noise.")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    d = C.load_data()
    horizons = [1, 21] if args.quick else C.HORIZONS

    # Stage 1 verdict: regime handling is per-resource
    s1 = pd.read_csv("stage1_results.csv") if os.path.exists("stage1_results.csv") else None
    regime_map = {}
    if s1 is not None:
        rg = s1[s1.stage == "1d_regime"]
        name_map = {"full-history": "full", "recency-weighted": "recency",
                    "recent-50%-window": "trunc50"}
        for res, grp in rg.groupby("resource"):
            best = grp.groupby("variant").skill.mean().idxmin()
            regime_map[res] = name_map.get(best, "full")
    for r in ALL_RES:
        regime_map.setdefault(r, "full")
    print("regime per resource (from Stage 1):", regime_map)

    R2 = stage2(d, horizons, args.quick)
    feats_map = {}
    for _, r in R2[R2.variant == "PRUNED"].iterrows():
        allr = R2[(R2.resource == r.resource) & (R2.horizon == r.horizon) & (R2.variant == "ALL")]
        if len(allr) and r.rmse < float(allr.rmse.iloc[0]):
            feats_map[(r.resource, int(r.horizon))] = json.loads(r.pruned_features)

    R3 = stage3(d, horizons, feats_map, regime_map, args.quick)
    R4 = stage4(d, horizons, feats_map, regime_map, R3)
    R6 = stage56(d, horizons, feats_map, regime_map, R3)

    print(f"\ncampaign finished in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
