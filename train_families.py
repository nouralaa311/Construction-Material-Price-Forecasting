"""Train EVERY (resource x horizon x model family) so the Streamlit app can offer a real
model-family selector — and so gaps are reported instead of silently falling back.

Saves one artifact per combination:  models/{resource}_h{h}__{family}.joblib
plus the coverage matrix in coverage_matrix.csv.

A combination that cannot be trained (too little data, convergence failure) is recorded as
FAILED with the reason. It is never silently replaced by another model — silent fallback is
exactly how an app ends up showing steel's model for every resource.

Every family gets identical treatment: walk-forward with a purge/embargo sized to the max
horizon, multi-seed where the model is stochastic, one held-out test evaluation, and a
Diebold-Mariano test against persistence.

Usage:
    python train_families.py                    # all families, all resources, all horizons
    python train_families.py --fast             # skip the deep (GRU) families
    python train_families.py --resources steel  # subset
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd

import h21_core as C
import stage_campaign as SC

warnings.filterwarnings("ignore")

TABULAR = ["ridge", "elasticnet", "rf", "extratrees", "xgboost", "lightgbm", "catboost"]
DEEP = ["gru", "bigru"]
SEEDS_STOCHASTIC = (42, 1, 7)
STOCHASTIC = {"rf", "extratrees", "xgboost", "lightgbm", "catboost", "gru", "bigru"}


# ---------------------------------------------------------------------------------------
# deep families
# ---------------------------------------------------------------------------------------
def train_deep(ctx, h, feats, arch, seed=C.SEED, regime="full"):
    """GRU / BiGRU. Returns (artifact_payload, walk_forward_DataFrame)."""
    import torch
    from train_all import GRUForecaster, make_sequences, gru_predict, train_gru

    X, Y, P = ctx["X"], ctx["Y"], ctx["P"]
    y = Y[f"y_{SC.TARGET_KIND}_h{h}"].values
    ytrue = Y[f"y_price_h{h}"].values
    p = dict(arch=arch, seq_len=14, hidden=64, layers=2 if arch == "bigru" else 1,
             dropout=0.15, lr=1e-3, wd=1e-4, batch=32, loss="huber", scaler="standard")

    rows = []
    for tr, va in ctx["folds"]:
        m = ~np.isnan(y[tr])
        tr_use = tr[m]
        if regime == "trunc50":
            tr_use = tr_use[-max(120, len(tr_use) // 2):]
        if len(tr_use) < 80:
            continue
        Xtr_df, (Xfull,) = C.impute_fold(X.iloc[tr_use][feats], [X[feats]], "ffill_median")
        fsc = C.SCALERS["standard"]().fit(Xtr_df.values)
        Xm = np.clip(np.nan_to_num(fsc.transform(Xfull.values).astype(np.float32)), -10, 10)
        ysc = C.SCALERS["standard"]().fit(y[tr_use].reshape(-1, 1))
        yz = np.full_like(y, np.nan, float)
        ok = ~np.isnan(y)
        yz[ok] = ysc.transform(y[ok].reshape(-1, 1)).ravel()
        Str, ytr_s, _ = make_sequences(Xm, yz, int(p["seq_len"]), tr_use)
        Sva, yva_s, kva = make_sequences(Xm, yz, int(p["seq_len"]), va)
        if len(Str) < 60 or len(Sva) < 10:
            continue
        mdl = train_gru(Str, ytr_s, Sva, yva_s, p)
        pred = ysc.inverse_transform(gru_predict(mdl, Sva).reshape(-1, 1)).ravel()
        pp = C.target_to_price(pred, P.values[kva], SC.TARGET_KIND)
        rows.append(C.evaluate(ytrue[kva], pp, P.values[kva], 0.0))
    S = pd.DataFrame(rows) if rows else pd.DataFrame([SC._INF])

    # final fit on TRAIN+VAL
    DEV = np.concatenate([ctx["tr"], ctx["va"]])
    m = ~np.isnan(y[DEV])
    dev_use = DEV[m]
    if regime == "trunc50":
        dev_use = dev_use[-max(120, len(dev_use) // 2):]
    Xd_df, (Xfull,) = C.impute_fold(X.iloc[dev_use][feats], [X[feats]], "ffill_median")
    fsc = C.SCALERS["standard"]().fit(Xd_df.values)
    Xm = np.clip(np.nan_to_num(fsc.transform(Xfull.values).astype(np.float32)), -10, 10)
    ysc = C.SCALERS["standard"]().fit(y[dev_use].reshape(-1, 1))
    yz = np.full_like(y, np.nan, float)
    ok = ~np.isnan(y)
    yz[ok] = ysc.transform(y[ok].reshape(-1, 1)).ravel()
    inner = int(len(dev_use) * 0.85)
    Str, ytr_s, _ = make_sequences(Xm, yz, int(p["seq_len"]), dev_use[:inner])
    Sva, yva_s, _ = make_sequences(Xm, yz, int(p["seq_len"]), dev_use[inner:])
    if len(Str) < 60 or len(Sva) < 5:
        raise RuntimeError("insufficient sequences for a final deep fit")
    mdl = train_gru(Str, ytr_s, Sva, yva_s, p)
    Ste, _, kte = make_sequences(Xm, np.zeros_like(yz), int(p["seq_len"]), ctx["te"])
    raw = ysc.inverse_transform(gru_predict(mdl, Ste).reshape(-1, 1)).ravel()
    pred_te = C.target_to_price(raw, P.values[kte], SC.TARGET_KIND)
    payload = dict(model=mdl.state_dict(), params=p, feature_scaler=fsc, target_scaler=ysc,
                   strategy="ffill_median", family="gru")
    return payload, S, (kte, pred_te)


def train_tabular(ctx, h, feats, family, seed=C.SEED, regime="full"):
    X, Y, P = ctx["X"], ctx["Y"], ctx["P"]
    y = Y[f"y_{SC.TARGET_KIND}_h{h}"].values
    S = SC.wf_score(family, ctx, h, feats, seed=seed, regime=regime)
    DEV = np.concatenate([ctx["tr"], ctx["va"]])
    m = ~np.isnan(y[DEV])
    dev_use = DEV[m]
    sw = None
    if regime == "trunc50":
        dev_use = dev_use[-max(120, len(dev_use) // 2):]
    elif regime == "recency":
        age = np.arange(len(dev_use))[::-1]
        sw = np.exp(-age / max(60.0, len(dev_use) / 4))
    strategy = "ffill_median" if family in SC.NEEDS_DENSE else "native_nan"
    Xtr, (Xte,) = C.impute_fold(X.iloc[dev_use][feats], [X.iloc[ctx["te"]][feats]], strategy)
    mdl = SC.make_model(family, None, seed)
    try:
        mdl.fit(Xtr, y[dev_use], sample_weight=sw)
    except TypeError:
        mdl.fit(Xtr, y[dev_use])
    pred_te = C.target_to_price(mdl.predict(Xte), P.values[ctx["te"]], SC.TARGET_KIND)
    payload = dict(model=mdl, params={}, strategy=strategy, family=family)
    return payload, S, (np.asarray(ctx["te"]), pred_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip GRU/BiGRU")
    ap.add_argument("--families", nargs="*", default=None,
                    help="train only these families (default: all)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not retrain a combination whose artifact already exists")
    ap.add_argument("--resources", nargs="*", default=list(C.RESOURCES))
    ap.add_argument("--horizons", nargs="*", type=int, default=C.HORIZONS)
    args = ap.parse_args()

    families = args.families or (TABULAR + ([] if args.fast else DEEP))
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    d = C.load_data()

    regime_map = {}
    if os.path.exists("stage1_results.csv"):
        s1 = pd.read_csv("stage1_results.csv")
        nm = {"full-history": "full", "recency-weighted": "recency", "recent-50%-window": "trunc50"}
        for res, grp in s1[s1.stage == "1d_regime"].groupby("resource"):
            regime_map[res] = nm.get(grp.groupby("variant").skill.mean().idxmin(), "full")
    for r in args.resources:
        regime_map.setdefault(r, "full")

    feats_map = {}
    if os.path.exists("stage2_results.csv"):
        s2 = pd.read_csv("stage2_results.csv")
        for _, r in s2[s2.variant == "PRUNED"].iterrows():
            a = s2[(s2.resource == r.resource) & (s2.horizon == r.horizon) & (s2.variant == "ALL")]
            if len(a) and r.rmse < float(a.rmse.iloc[0]) and isinstance(r.pruned_features, str):
                feats_map[(r.resource, int(r.horizon))] = json.loads(r.pruned_features)

    cov, t0 = [], time.time()
    for res in args.resources:
        ctx = SC.ctx_for(d, res)
        regime = regime_map.get(res, "full")
        for h in args.horizons:
            feats = feats_map.get((res, h)) or list(ctx["X"].columns)
            nb_wf = SC.naive_wf(ctx, h)
            ytrue_te = ctx["Y"][f"y_price_h{h}"].values
            base_all = ctx["P"].values
            for fam in families:
                rec = dict(resource=res, horizon=h, family=fam, regime=regime, status="",
                           reason="", wf_rmse=np.nan, wf_skill=np.nan, test_rmse=np.nan,
                           test_r2=np.nan, test_diracc=np.nan, naive_rmse=np.nan, dm_p=np.nan,
                           beats_naive=False, seeds=1)
                stem_path = os.path.join(C.MODEL_DIR, f"{res}_h{h}__{fam}.joblib")
                if args.skip_existing and os.path.exists(stem_path):
                    rec.update(status="SKIPPED", reason="artifact already exists")
                    cov.append(rec)
                    print(f"  {res:<10} h={h:>2} {fam:<11} skip (exists)")
                    continue
                try:
                    seeds = SEEDS_STOCHASTIC if fam in STOCHASTIC else (C.SEED,)
                    wfs, payload, te_out = [], None, None
                    for sd in seeds:
                        if fam in DEEP:
                            gf = feats[:40]
                            pl, S, out = train_deep(ctx, h, gf, "bigru" if fam == "bigru" else "gru",
                                                    sd, regime)
                            pl["family"] = fam
                            pl["gru_features"] = gf
                        else:
                            pl, S, out = train_tabular(ctx, h, feats, fam, sd, regime)
                        wfs.append(float(S.RMSE.mean()))
                        if payload is None:
                            payload, te_out = pl, out
                    kte, pred_te = te_out
                    yv, bv = ytrue_te[kte], base_all[kte]
                    ok = ~np.isnan(yv)
                    if ok.sum() < 5:
                        raise RuntimeError("no usable test rows")
                    ev = C.evaluate(yv[ok], pred_te[ok], bv[ok], 0.0, fam, h)
                    nv = C.evaluate(yv[ok], bv[ok], bv[ok], 0.0, "naive", h)
                    dm_s, dm_p = C.diebold_mariano(yv[ok], pred_te[ok], bv[ok])
                    wf_mean = float(np.mean(wfs))

                    art = dict(resource=res, horizon=h, target_kind=SC.TARGET_KIND,
                               winner=fam, family=fam, regime=regime,
                               strategy=payload["strategy"], dead_zone=0.0,
                               features=feats, gru_features=payload.get("gru_features", feats),
                               model=payload["model"], params=payload["params"],
                               **{k: payload[k] for k in ("feature_scaler", "target_scaler")
                                  if k in payload},
                               accepted=bool(ev["RMSE"] < nv["RMSE"]),
                               metrics=dict(val_rmse=wf_mean, val_rmse_std=float(np.std(wfs)),
                                            val_r2=float("nan"), val_diracc=None,
                                            wf_skill=wf_mean / nb_wf,
                                            test_rmse=ev["RMSE"], test_mae=ev["MAE"],
                                            test_r2=ev["R2"], test_mape=ev["MAPE%"],
                                            test_diracc=ev["DirAcc"], test_coverage=ev["Coverage"],
                                            test_n_dir=ev["N_dir"], test_p_value=ev["p_value"],
                                            test_precision=ev["Precision"], test_recall=ev["Recall"],
                                            test_f1=ev["F1"], test_cm=ev["cm"],
                                            test_diracc_full=ev["DirAcc"],
                                            test_n_dir_full=ev["N_dir"],
                                            best_any_diracc=ev["DirAcc"], best_any_dz=0.0,
                                            best_any_coverage=ev["Coverage"],
                                            naive_rmse=nv["RMSE"], dm_stat=dm_s, dm_p=dm_p,
                                            beats_naive=bool(ev["RMSE"] < nv["RMSE"]),
                                            beats_naive_significant=bool(dm_p < 0.05 and dm_s < 0),
                                            n_seeds=len(seeds)),
                               meta=dict(resource=res, label=C.RESOURCES[res]["label"],
                                         unit=C.RESOURCES[res]["unit"], horizon=h,
                                         approx_days=C.HORIZON_DAYS[h],
                                         target_kind=SC.TARGET_KIND, winner=fam, family=fam,
                                         regime=regime, strategy=payload["strategy"],
                                         dead_zone=0.0, accepted=bool(ev["RMSE"] < nv["RMSE"]),
                                         n_samples=len(ctx["X"]), n_features=len(feats),
                                         train_start=str(ctx["D"].iloc[ctx["tr"][0]].date()),
                                         train_end=str(ctx["D"].iloc[ctx["tr"][-1]].date()),
                                         test_start=str(ctx["D"].iloc[ctx["te"][0]].date()),
                                         test_end=str(ctx["D"].iloc[ctx["te"][-1]].date()),
                                         n_train=len(ctx["tr"]), n_val=len(ctx["va"]),
                                         n_test=len(ctx["te"]), top_features=feats[:15]))
                    stem = f"{res}_h{h}__{fam}"
                    joblib.dump(art, os.path.join(C.MODEL_DIR, f"{stem}.joblib"))
                    with open(os.path.join(C.MODEL_DIR, f"{stem}.meta.json"), "w") as fh:
                        json.dump({"meta": art["meta"], "metrics": art["metrics"],
                                   "params": {}, "features": feats}, fh, indent=2, default=str)
                    rec.update(status="TRAINED", wf_rmse=wf_mean, wf_skill=wf_mean / nb_wf,
                               test_rmse=ev["RMSE"], test_r2=ev["R2"], test_diracc=ev["DirAcc"],
                               naive_rmse=nv["RMSE"], dm_p=dm_p,
                               beats_naive=bool(ev["RMSE"] < nv["RMSE"]), seeds=len(seeds))
                except Exception as e:
                    rec.update(status="FAILED", reason=f"{type(e).__name__}: {e}"[:160])
                cov.append(rec)
                flag = ("ok " if rec["status"] == "TRAINED" else "ERR")
                print(f"  {res:<10} h={h:>2} {fam:<11} {flag} "
                      + (f"wf={rec['wf_rmse']:>9.2f} test={rec['test_rmse']:>9.2f} "
                         f"{'BEATS' if rec['beats_naive'] else 'loses'}"
                         if rec["status"] == "TRAINED" else rec["reason"]))

    COV = pd.DataFrame(cov)
    COV.to_csv("coverage_matrix.csv", index=False)
    print("\n" + "=" * 100)
    print("COVERAGE MATRIX".center(100))
    print("=" * 100)
    piv = COV.pivot_table(index=["resource", "horizon"], columns="family", values="status",
                          aggfunc="first")
    print(piv.to_string())
    print(f"\nTRAINED {int((COV.status=='TRAINED').sum())} / {len(COV)}   "
          f"FAILED {int((COV.status=='FAILED').sum())}")
    if (COV.status == "FAILED").any():
        print("\nFAILURES (never silently substituted):")
        print(COV[COV.status == "FAILED"][["resource", "horizon", "family", "reason"]]
              .to_string(index=False))
    ok = COV[COV.status == "TRAINED"]
    print(f"\nbeat persistence on TEST: {int(ok.beats_naive.sum())} / {len(ok)}")
    print("\nmean test RMSE skill by family (test_rmse / naive_rmse; <1 beats persistence):")
    ok = ok.assign(skill=ok.test_rmse / ok.naive_rmse)
    print(ok.pivot_table(index="family", columns="resource", values="skill",
                         aggfunc="mean").round(3).to_string())
    print(f"\nelapsed {time.time()-t0:.0f}s -> coverage_matrix.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
