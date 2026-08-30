"""Diagnose the two chart problems empirically.

PROBLEM 1 — vertical spikes in the in-arrears model line.
  Walks candidate causes (a)-(e) in order and reports which one it actually is,
  rather than guessing.

PROBLEM 2 — the model line lagging the actual price (persistence signature).
  Cross-correlates prediction vs actual at a range of lags. If the peak sits near +h
  rather than 0, the model is reproducing persistence, not forecasting.

Usage:  python diagnose_charts.py [resource] [horizon]
"""
from __future__ import annotations

import sys
import warnings

import joblib
import numpy as np
import pandas as pd

import h21_core as C

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


def in_arrears(artifact, d):
    """Reproduce the app's in-arrears computation, returning intermediates for inspection."""
    res, h, kind = artifact["resource"], artifact["horizon"], artifact["target_kind"]
    X, Y, P, D = C.build_dataset(d, res)
    n = len(X)
    fam = artifact.get("family", artifact["winner"])
    if fam != "gru" and hasattr(artifact["model"], "predict"):
        feats = [f for f in artifact["features"] if f in X.columns]
        Xi, _ = C.impute_fold(X[feats], [], artifact["strategy"])
        raw = artifact["model"].predict(Xi)
        src = np.arange(n)
    else:
        from train_all import GRUForecaster, make_sequences, gru_predict
        gf = [f for f in artifact["gru_features"] if f in X.columns]
        _, (Xfull,) = C.impute_fold(X[gf], [X[gf]], "ffill_median")
        Xm = np.clip(np.nan_to_num(
            artifact["feature_scaler"].transform(Xfull.values).astype(np.float32)), -10, 10)
        p = artifact["params"]
        mdl = GRUForecaster(Xm.shape[1], int(p["hidden"]), int(p["layers"]),
                            float(p["dropout"]), p["arch"])
        mdl.load_state_dict(artifact["model"])
        mdl.eval()
        S, _, keep = make_sequences(Xm, np.zeros(n), int(p["seq_len"]), np.arange(n))
        raw = artifact["target_scaler"].inverse_transform(
            gru_predict(mdl, S).reshape(-1, 1)).ravel()
        src = keep
    anchor = P.values[src]
    pred = C.target_to_price(raw, anchor, kind)
    tgt = src + h
    ok = tgt < n
    return pd.DataFrame({
        "origin_date": D.iloc[src[ok]].values,
        "target_date": D.iloc[tgt[ok]].values,
        "anchor_price": anchor[ok],
        "raw_pred": raw[ok],
        "model_price": pred[ok],
        "actual_at_target": P.values[tgt[ok]],
        "actual_at_origin": P.values[src[ok]],
    })


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else "steel"
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 21
    art = joblib.load(f"{C.MODEL_DIR}/{res}_h{h}.joblib")
    d = C.load_data()
    T = in_arrears(art, d)

    print("=" * 100)
    print(f"CHART DIAGNOSTICS — {res} h={h}  (family={art.get('family', art['winner'])}, "
          f"target={art['target_kind']})".center(100))
    print("=" * 100)

    # ---------------------------------------------------------------- PROBLEM 1
    print("\nPROBLEM 1 — SPIKES: checking candidate causes")
    m = T["model_price"].values
    a = T["actual_at_target"].values
    dm = np.diff(m)
    da = np.diff(a)
    vol_ratio = np.nanstd(dm) / np.nanstd(da)
    print(f"  step-to-step volatility: model={np.nanstd(dm):,.2f}  actual={np.nanstd(da):,.2f}  "
          f"ratio={vol_ratio:.2f}x")

    # (b) duplicate / non-monotonic dates
    dup = int(T.target_date.duplicated().sum())
    mono = bool(pd.Series(T.target_date).is_monotonic_increasing)
    print(f"  (b) duplicate target dates = {dup} | monotonic = {mono} -> "
          f"{'PROBLEM' if dup or not mono else 'ok'}")

    # (a) NaN / sentinel values
    n_nan = int(np.isnan(m).sum())
    n_zero = int((m == 0).sum())
    print(f"  (a) NaN in model line = {n_nan} | exact zeros = {n_zero} -> "
          f"{'PROBLEM' if n_nan or n_zero else 'ok'}")

    # (c) in-arrears alignment spot-check on 3 rows
    print("  (c) alignment spot-check (target_date must be origin_date + h STEPS):")
    X, Y, P, D = C.build_dataset(d, res)
    okc = True
    for i in (10, len(T) // 2, len(T) - 5):
        o_i = int(np.where(D.values == T.origin_date.iloc[i])[0][0])
        t_i = int(np.where(D.values == T.target_date.iloc[i])[0][0])
        good = (t_i - o_i) == h
        okc &= good
        print(f"        row {i}: origin idx {o_i} -> target idx {t_i} (delta={t_i-o_i}) "
              f"{'ok' if good else 'MISALIGNED'}")

    # (d) inverse-transform anchor check
    print("  (d) inverse-transform anchor:")
    if art["target_kind"] == "logret":
        recon = T.anchor_price.values * np.exp(T.raw_pred.values)
        anchor_ok = np.allclose(recon, T.model_price.values, rtol=1e-9)
        same_anchor = T.anchor_price.nunique() <= 1
        print(f"        each prediction anchored to its OWN origin price = "
              f"{not same_anchor} (unique anchors: {T.anchor_price.nunique()})")
        print(f"        reconstruction matches = {anchor_ok}")
        big = np.abs(T.raw_pred.values) > 0.05
        print(f"        |predicted logret| > 5%: {int(big.sum())} rows "
              f"({big.mean():.1%}) -> these ARE the spikes if large")
        print(f"        raw prediction range: [{T.raw_pred.min():+.4f}, {T.raw_pred.max():+.4f}] "
              f"(actual h-step logret sd = {np.nanstd(np.log(a/T.actual_at_origin.values)):.4f})")

    # (e) calendar gaps
    gaps = pd.Series(T.target_date).diff().dt.days.dropna()
    print(f"  (e) target-date gaps: median={gaps.median():.0f}d max={gaps.max():.0f}d "
          f"| >7d gaps = {int((gaps > 7).sum())}")

    # locate the worst spikes and characterise them
    print("\n  WORST 5 STEP-CHANGES IN THE MODEL LINE:")
    idx = np.argsort(-np.abs(dm))[:5]
    for i in sorted(idx):
        print(f"    {pd.Timestamp(T.target_date.iloc[i]).date()} -> "
              f"{pd.Timestamp(T.target_date.iloc[i+1]).date()}  "
              f"model {m[i]:>10,.0f} -> {m[i+1]:>10,.0f} (delta {dm[i]:>+9,.0f}) | "
              f"anchor {T.anchor_price.iloc[i]:>9,.0f} -> {T.anchor_price.iloc[i+1]:>9,.0f} "
              f"(delta {T.anchor_price.iloc[i+1]-T.anchor_price.iloc[i]:>+8,.0f}) | "
              f"raw {T.raw_pred.iloc[i]:+.4f} -> {T.raw_pred.iloc[i+1]:+.4f}")

    anchor_steps = np.diff(T.anchor_price.values)
    corr_spike_anchor = np.corrcoef(np.abs(dm), np.abs(anchor_steps))[0, 1]
    print(f"\n  corr(|model step|, |anchor step|) = {corr_spike_anchor:.3f}")
    print("    -> high correlation means the spikes come from the ANCHOR price series "
          "(the real quote jumping), not from the model's own output.")

    # ---------------------------------------------------------------- PROBLEM 2
    print("\n" + "-" * 100)
    print("PROBLEM 2 — LAG / PERSISTENCE SIGNATURE")
    print("-" * 100)
    s_m = pd.Series(m).astype(float)
    s_a = pd.Series(a).astype(float)
    print("  cross-correlation of model prediction vs ACTUAL price at lag k")
    print("  (model shifted forward by k; peak at k=0 = genuine forecast, peak near "
          f"k={h} = persistence)")
    # Scan NEGATIVE lags too. If the model reproduces persistence, pred(T) ~ price(T-h), so the
    # model series is a DELAYED copy and the peak sits at k = -h. Scanning only k>=0 cannot
    # detect that at all - it just shows monotonic decay and looks like "peak at 0".
    best = (None, -9)
    span = min(30, len(s_a) // 4)
    for k in range(-span, span + 1):
        c = s_m.shift(k).corr(s_a)
        if np.isfinite(c) and c > best[1]:
            best = (k, c)
        if k % 3 == 0 or abs(k) == h:
            tag = "   <-- +horizon" if k == h else ("   <-- -horizon (persistence)" if k == -h else "")
            print(f"    lag {k:>4}: corr = {c:+.4f}{tag}")
    print(f"\n  PEAK correlation at lag {best[0]} (corr={best[1]:.4f})")
    verdict = ("PERSISTENCE CONFIRMED - the model is reproducing a lagged copy of the price"
               if best[0] is not None and abs(best[0] - h) <= max(2, h // 4)
               else "peak near lag 0 - not a simple lagged copy")
    print(f"  VERDICT: {verdict}")

    # correlation of the model's own PREDICTED CHANGE with the ACTUAL change
    pred_chg = T.model_price.values - T.actual_at_origin.values
    act_chg = T.actual_at_target.values - T.actual_at_origin.values
    ok2 = np.isfinite(pred_chg) & np.isfinite(act_chg)
    r_chg = np.corrcoef(pred_chg[ok2], act_chg[ok2])[0, 1]
    print(f"\n  corr(predicted CHANGE, actual CHANGE) = {r_chg:+.4f}")
    print("    This is the number that matters: predicting the LEVEL well is easy "
          "(persistence does it), predicting the CHANGE is the actual forecasting task.")

    # CRITICAL: split in-sample from out-of-sample. Metrics over all history are dominated by
    # rows the model was fitted on and look far better than the model actually is.
    test_start = pd.Timestamp(art["meta"]["test_start"])
    is_test = pd.Series(T.target_date).ge(test_start).values
    print(f"\n  test period begins {test_start.date()} "
          f"({int(is_test.sum())} of {len(T)} plotted points are out-of-sample)")

    for label, mask in (("IN-SAMPLE (model was trained on these rows)", ~is_test),
                        ("OUT-OF-SAMPLE (the only honest number)", is_test)):
        if mask.sum() < 10:
            continue
        mm, aa, oo = m[mask], a[mask], T.actual_at_origin.values[mask]
        rm = float(np.sqrt(np.nanmean((aa - mm) ** 2)))
        rp = float(np.sqrt(np.nanmean((aa - oo) ** 2)))
        pc, ac = mm - oo, aa - oo
        good = np.isfinite(pc) & np.isfinite(ac)
        rc = np.corrcoef(pc[good], ac[good])[0, 1] if good.sum() > 5 else float("nan")
        # lag of peak correlation within this segment
        sm2, sa2 = pd.Series(mm), pd.Series(aa)
        _sp = min(25, len(sa2) // 4)
        lags = {k: sm2.shift(k).corr(sa2) for k in range(-_sp, _sp + 1)}
        pk = max(lags, key=lambda k: (lags[k] if np.isfinite(lags[k]) else -9))
        print(f"\n  [{label}]  n={int(mask.sum())}")
        print(f"    RMSE model={rm:>10,.2f}  persistence={rp:>10,.2f}  "
              f"-> {(1-rm/rp)*100:+.2f}% ({'better' if rm < rp else 'WORSE'})")
        print(f"    corr(predicted change, actual change) = {rc:+.4f}")
        print(f"    peak cross-correlation lag = {pk} "
              f"({"PERSISTENCE-LIKE" if abs(abs(pk) - h) <= max(2, h // 4) else "not a lagged copy"})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
