"""Retrain the campaign's winning configuration per (resource, horizon) and save artifacts
that the Streamlit app consumes.

Selection is taken from stage6_final.csv (which already applied Diebold-Mariano + Benjamini-
Hochberg). The TEST set is touched exactly once here, at the end, after everything is frozen.

Honesty: a resource/horizon that did NOT clear Stage 6 still gets an artifact so the app can
display it, but its metadata carries `accepted=False` and the app shows the "does not beat
naive" banner. We do not quietly promote a loser.

Usage:  python finalize_models.py
"""
from __future__ import annotations

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

import h21_core as C
import stage_campaign as SC

warnings.filterwarnings("ignore")


def main() -> int:
    if not os.path.exists("stage6_final.csv"):
        print("stage6_final.csv missing — run `python stage_campaign.py` first.")
        return 1
    S6 = pd.read_csv("stage6_final.csv")
    S2 = pd.read_csv("stage2_results.csv") if os.path.exists("stage2_results.csv") else None

    feats_map = {}
    if S2 is not None:
        for _, r in S2[S2.variant == "PRUNED"].iterrows():
            allr = S2[(S2.resource == r.resource) & (S2.horizon == r.horizon) & (S2.variant == "ALL")]
            if len(allr) and r.rmse < float(allr.rmse.iloc[0]) and isinstance(r.pruned_features, str):
                feats_map[(r.resource, int(r.horizon))] = json.loads(r.pruned_features)

    d = C.load_data()
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    written = []

    for _, row in S6.iterrows():
        res, h = row.resource, int(row.horizon)
        fam, regime = row.family, row.regime
        ctx = SC.ctx_for(d, res)
        X, Y, P, D = ctx["X"], ctx["Y"], ctx["P"], ctx["D"]
        feats = feats_map.get((res, h)) or list(X.columns)
        tr, va, te = ctx["tr"], ctx["va"], ctx["te"]
        DEV = np.concatenate([tr, va])

        y = Y[f"y_{SC.TARGET_KIND}_h{h}"].values
        m = ~np.isnan(y[DEV])
        dev_use = DEV[m]
        sw = None
        if regime == "trunc50":
            dev_use = dev_use[-max(120, len(dev_use) // 2):]
        elif regime == "recency":
            age = np.arange(len(dev_use))[::-1]
            sw = np.exp(-age / max(60.0, len(dev_use) / 4))

        strategy = "ffill_median" if fam in SC.NEEDS_DENSE else "native_nan"
        Xtr, (Xte,) = C.impute_fold(X.iloc[dev_use][feats], [X.iloc[te][feats]], strategy)
        model = SC.make_model(fam)
        try:
            model.fit(Xtr, y[dev_use], sample_weight=sw)
        except TypeError:
            model.fit(Xtr, y[dev_use])

        ytrue_te = Y[f"y_price_h{h}"].values[te]
        base_te = P.values[te]
        ok = ~np.isnan(ytrue_te)
        pred_te = C.target_to_price(model.predict(Xte), base_te, SC.TARGET_KIND)

        # frozen dead zone chosen on VALIDATION OOF, never on test
        _, oof = SC.wf_score(fam, ctx, h, feats, regime=regime, return_oof=True)
        F = SC.oof_frame(oof)
        best_dz = 0.0
        if len(F):
            ob = P.values[F.index.values]
            cand = []
            for dz in C.CFG.dead_zones:
                s = C.directional_stats(F["true"].values, F["pred"].values, ob, dz)
                if s["coverage"] >= C.CFG.coverage_floor and not pd.isna(s["dir_acc"]):
                    cand.append((s["dir_acc"], dz))
            best_dz = max(cand)[1] if cand else 0.0

        test_row = C.evaluate(ytrue_te[ok], pred_te[ok], base_te[ok], best_dz, fam, h)
        full_row = C.evaluate(ytrue_te[ok], pred_te[ok], base_te[ok], 0.0, fam, h)
        naive_row = C.evaluate(ytrue_te[ok], base_te[ok], base_te[ok], best_dz, "naive", h)
        dm_stat, dm_p = C.diebold_mariano(ytrue_te[ok], pred_te[ok], base_te[ok])
        best_any = max((C.directional_stats(ytrue_te[ok], pred_te[ok], base_te[ok], dz)
                        for dz in C.CFG.dead_zones),
                       key=lambda s: -1 if s["dir_acc"] is None or pd.isna(s["dir_acc"]) else s["dir_acc"])

        art = dict(
            resource=res, horizon=h, target_kind=SC.TARGET_KIND, winner=fam, family=fam,
            regime=regime, strategy=strategy, dead_zone=best_dz,
            features=feats, gru_features=feats, model=model,
            params={}, accepted=bool(row.get("bh_significant", False)),
            metrics=dict(
                val_rmse=float(row.rmse_mean), val_rmse_std=float(row.rmse_std),
                val_r2=float("nan"), val_diracc=float(row.diracc) if pd.notna(row.diracc) else None,
                test_rmse=test_row["RMSE"], test_mae=test_row["MAE"], test_r2=test_row["R2"],
                test_mape=test_row["MAPE%"], test_diracc=test_row["DirAcc"],
                test_coverage=test_row["Coverage"], test_n_dir=test_row["N_dir"],
                test_p_value=test_row["p_value"], test_precision=test_row["Precision"],
                test_recall=test_row["Recall"], test_f1=test_row["F1"], test_cm=test_row["cm"],
                test_diracc_full=full_row["DirAcc"], test_n_dir_full=full_row["N_dir"],
                best_any_diracc=best_any["dir_acc"], best_any_dz=best_any["dead_zone"],
                best_any_coverage=best_any["coverage"],
                naive_rmse=naive_row["RMSE"], dm_stat=dm_stat, dm_p=dm_p,
                beats_naive=bool(test_row["RMSE"] < naive_row["RMSE"]),
                beats_naive_significant=bool(dm_p < 0.05 and dm_stat < 0),
                wf_dm_p=float(row.dm_p), wf_skill=float(row.skill),
                conformal_q90=float(row.conformal_q90) if pd.notna(row.conformal_q90) else None,
                conformal_coverage=float(row.conformal_coverage) if pd.notna(row.conformal_coverage) else None,
                bh_significant=bool(row.get("bh_significant", False)),
                stage0_verdict=row.get("signal_group", ""),
            ),
            meta=dict(resource=res, label=C.RESOURCES[res]["label"], unit=C.RESOURCES[res]["unit"],
                      horizon=h, approx_days=C.HORIZON_DAYS[h], target_kind=SC.TARGET_KIND,
                      winner=fam, family=fam, regime=regime, strategy=strategy,
                      dead_zone=best_dz, accepted=bool(row.get("bh_significant", False)),
                      n_samples=len(X), n_features=len(feats),
                      train_start=str(D.iloc[tr[0]].date()), train_end=str(D.iloc[tr[-1]].date()),
                      test_start=str(D.iloc[te[0]].date()), test_end=str(D.iloc[te[-1]].date()),
                      n_train=len(tr), n_val=len(va), n_test=len(te),
                      top_features=feats[:15]),
        )
        path = os.path.join(C.MODEL_DIR, f"{res}_h{h}.joblib")
        joblib.dump(art, path)
        with open(os.path.join(C.MODEL_DIR, f"{res}_h{h}.meta.json"), "w") as fh:
            json.dump({"meta": art["meta"], "metrics": art["metrics"],
                       "params": art["params"], "features": art["features"]},
                      fh, indent=2, default=str)
        written.append(dict(resource=res, horizon=h, family=fam, regime=regime,
                            test_rmse=test_row["RMSE"], naive_rmse=naive_row["RMSE"],
                            test_r2=test_row["R2"], test_diracc=test_row["DirAcc"],
                            test_coverage=test_row["Coverage"], dm_p=dm_p,
                            beats_naive=test_row["RMSE"] < naive_row["RMSE"],
                            accepted=bool(row.get("bh_significant", False))))
        print(f"  {res:<10} h={h:>2} {fam:<11} {regime:<8} test RMSE={test_row['RMSE']:>9.2f} "
              f"(naive {naive_row['RMSE']:>9.2f}) accepted={art['accepted']}")

    LB = pd.DataFrame(written)
    LB.to_csv("leaderboard.csv", index=False)
    print("\n" + "=" * 96)
    print("FINAL LEADERBOARD (test set, evaluated once)".center(96))
    print("=" * 96)
    print(LB.round(4).to_string(index=False))
    print(f"\nbeat naive on test: {int(LB.beats_naive.sum())}/{len(LB)}   "
          f"accepted by Stage 6: {int(LB.accepted.sum())}/{len(LB)}")
    print("wrote leaderboard.csv + models/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
