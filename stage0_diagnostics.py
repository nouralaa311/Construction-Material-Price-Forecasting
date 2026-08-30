"""STAGE 0 — Establish the ceiling and the floor before optimizing.

Answers, per resource and per horizon, the only question that should drive effort allocation:
"is there exploitable structure here at all, or is the naive baseline already near-optimal?"

Nothing here trains a model. These are properties of the DATA.

Tests
-----
* Naive / random-walk baseline   -> the bar every later stage must clear.
* ACF of returns + Ljung-Box     -> is there linear autocorrelation to exploit?
* Hurst exponent (R/S)           -> >0.5 trending (momentum), <0.5 mean-reverting, ~0.5 random.
* Lo-MacKinlay variance ratio    -> formal random-walk test at each horizon q, heteroskedasticity-
                                    robust z-stat. Rejecting RW is direct evidence of structure.
* Permutation entropy (Bandt-Pompe) -> ordinal complexity; ~1.0 means near-random.
* Drift-to-noise per horizon     -> |mean(r_h)| / sd(r_h); governs achievable directional accuracy.

Output: stage0_diagnostics.csv + a headroom verdict per resource/horizon.

Usage:  python stage0_diagnostics.py
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import scipy.stats as st

import h21_core as C

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


# =======================================================================================
# Predictability statistics
# =======================================================================================
def hurst_rs(x: np.ndarray, min_chunk: int = 16) -> float:
    """Hurst exponent via rescaled-range (R/S) analysis on log-returns."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 64:
        return np.nan
    sizes, rs = [], []
    size = min_chunk
    while size <= n // 2:
        chunks = n // size
        vals = []
        for i in range(chunks):
            seg = x[i * size:(i + 1) * size]
            z = seg - seg.mean()
            cum = np.cumsum(z)
            R = cum.max() - cum.min()
            S = seg.std(ddof=1)
            if S > 0:
                vals.append(R / S)
        if vals:
            sizes.append(size)
            rs.append(np.mean(vals))
        size *= 2
    if len(sizes) < 3:
        return np.nan
    return float(np.polyfit(np.log(sizes), np.log(rs), 1)[0])


def variance_ratio(x: np.ndarray, q: int):
    """Lo-MacKinlay variance-ratio test, heteroskedasticity-robust.

    VR(q) = Var(q-period return) / (q * Var(1-period return)).
    VR = 1 under a random walk. z is the robust test statistic; |z| > 1.96 rejects RW at 5%.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < q * 4 or q < 2:
        return np.nan, np.nan, np.nan
    mu = x.mean()
    var1 = np.sum((x - mu) ** 2) / (n - 1)
    if var1 <= 0:
        return np.nan, np.nan, np.nan
    # overlapping q-period sums
    cs = np.cumsum(np.insert(x, 0, 0.0))
    q_sums = cs[q:] - cs[:-q]
    m = q * (n - q + 1) * (1 - q / n)
    # Lo-MacKinlay's normalizer m already carries the q factor, so sigma^2(q) is directly
    # comparable to sigma^2(1). Dividing by q again here would be double-counting and makes
    # VR decay like 1/q even for a pure random walk (verified against synthetic series).
    varq = np.sum((q_sums - q * mu) ** 2) / m if m > 0 else np.nan
    vr = varq / var1
    # robust variance of VR (Lo-MacKinlay heteroskedasticity-consistent)
    theta = 0.0
    dev2 = (x - mu) ** 2
    denom = dev2.sum() ** 2
    for j in range(1, q):
        num = np.sum(dev2[j:] * dev2[:-j]) * n
        delta = num / denom if denom > 0 else 0.0
        theta += (2 * (q - j) / q) ** 2 * delta
    # Lo-MacKinlay: sqrt(T)*(VR-1)/sqrt(theta) ~ N(0,1). The sqrt(T) is essential - without it
    # the statistic is ~1/sqrt(T) too small and NOTHING ever looks significant.
    z = math.sqrt(n) * (vr - 1) / math.sqrt(theta) if theta > 0 else np.nan
    p = 2 * (1 - st.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return float(vr), float(z), float(p)


def permutation_entropy(x: np.ndarray, order: int = 4) -> float:
    """Normalized Bandt-Pompe permutation entropy. 1.0 = maximally random ordinal structure."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < order + 10:
        return np.nan
    perms = {p: 0 for p in itertools.permutations(range(order))}
    for i in range(n - order + 1):
        perms[tuple(np.argsort(x[i:i + order]))] += 1
    counts = np.array([c for c in perms.values() if c > 0], float)
    pk = counts / counts.sum()
    return float(-np.sum(pk * np.log(pk)) / np.log(math.factorial(order)))


def ljung_box(x: np.ndarray, lags: int = 10):
    """Ljung-Box Q test for autocorrelation up to `lags`."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lags + 10:
        return np.nan, np.nan
    xc = x - x.mean()
    denom = np.sum(xc ** 2)
    q = 0.0
    for k in range(1, lags + 1):
        r = np.sum(xc[k:] * xc[:-k]) / denom
        q += r * r / (n - k)
    q *= n * (n + 2)
    return float(q), float(1 - st.chi2.cdf(q, lags))


# =======================================================================================
def main() -> int:
    d = C.load_data()
    rows, series_rows = [], []

    print("=" * 108)
    print("STAGE 0 — INTRINSIC PREDICTABILITY & BASELINE FLOOR".center(108))
    print("=" * 108)

    for res in C.RESOURCES:
        X, Y, P, D = C.build_dataset(d, res)
        p = P.values.astype(float)
        lr = np.diff(np.log(p))
        lr = lr[np.isfinite(lr)]

        H = hurst_rs(lr)
        PE = permutation_entropy(lr, 4)
        lb_q, lb_p = ljung_box(lr, 10)
        zero_frac = float(np.mean(np.abs(lr) < 1e-12))

        series_rows.append(dict(resource=res, n_obs=len(p), n_returns=len(lr),
                                ret_std=float(lr.std()), zero_move_frac=zero_frac,
                                hurst=H, perm_entropy=PE, ljung_box_p=lb_p))

        print(f"\n{'-' * 108}")
        print(f"{C.RESOURCES[res]['label']}  ({res})   n={len(p)} observations")
        print(f"{'-' * 108}")
        print(f"  return sd = {lr.std():.4%} | zero-move days = {zero_frac:.1%}")
        print(f"  Hurst (R/S)          = {H:.3f}   "
              f"({'trending/momentum' if H > 0.55 else 'mean-reverting' if H < 0.45 else 'random-walk-like'})")
        print(f"  Permutation entropy  = {PE:.4f}   "
              f"({'near-random' if PE > 0.98 else 'some ordinal structure'})")
        print(f"  Ljung-Box(10) p      = {lb_p:.4g}   "
              f"({'autocorrelation present' if lb_p < 0.05 else 'no linear autocorrelation'})")

        for h in C.HORIZONS:
            yt = Y[f"y_price_h{h}"].values
            ok = ~np.isnan(yt)
            naive_rmse = float(np.sqrt(np.mean((yt[ok] - p[ok]) ** 2)))
            naive_mae = float(np.mean(np.abs(yt[ok] - p[ok])))
            rh = Y[f"y_logret_h{h}"].values[ok]
            rh = rh[np.isfinite(rh)]
            drift_noise = float(abs(rh.mean()) / rh.std()) if len(rh) > 5 and rh.std() > 0 else np.nan
            vr, vz, vp = variance_ratio(lr, max(2, h))

            # headroom verdict -------------------------------------------------------
            evidence = 0
            if np.isfinite(vp) and vp < 0.05:
                evidence += 2                     # formal RW rejection is the strongest signal
            if np.isfinite(lb_p) and lb_p < 0.05:
                evidence += 1
            if np.isfinite(H) and abs(H - 0.5) > 0.08:
                evidence += 1
            if np.isfinite(PE) and PE < 0.985:
                evidence += 1
            verdict = ("REAL SIGNAL" if evidence >= 3 else
                       "WEAK SIGNAL" if evidence == 2 else
                       "LITTLE/NO SIGNAL")
            budget = ("meaningful headroom" if evidence >= 3 else
                      "modest headroom" if evidence == 2 else
                      "naive baseline likely near-optimal")

            rows.append(dict(resource=res, horizon=h, naive_rmse=naive_rmse, naive_mae=naive_mae,
                             drift_to_noise=drift_noise, vr=vr, vr_z=vz, vr_p=vp,
                             hurst=H, perm_entropy=PE, ljung_box_p=lb_p,
                             evidence_score=evidence, verdict=verdict, headroom=budget))

            print(f"    h={h:>2}: naive RMSE={naive_rmse:>10.2f}  drift/noise={drift_noise:>6.3f}  "
                  f"VR({h})={vr:>5.2f} (p={vp:>7.4f})  -> {verdict:<17} [{budget}]")

    LB = pd.DataFrame(rows)
    LB.to_csv("stage0_diagnostics.csv", index=False)
    pd.DataFrame(series_rows).to_csv("stage0_series_stats.csv", index=False)

    print("\n" + "=" * 108)
    print("EXPECTED-IMPROVEMENT BUDGET".center(108))
    print("=" * 108)
    piv = LB.pivot(index="resource", columns="horizon", values="verdict")
    print(piv.to_string())
    print("\nby verdict:")
    print(LB.verdict.value_counts().to_string())

    real = LB[LB.verdict == "REAL SIGNAL"]
    none = LB[LB.verdict == "LITTLE/NO SIGNAL"]
    print(f"\nSPEND EFFORT HERE ({len(real)} combos):")
    for _, r in real.iterrows():
        print(f"  {r.resource:<10} h={int(r.horizon):<3} VR p={r.vr_p:.4f}  Hurst={r.hurst:.3f}")
    print(f"\nDO NOT EXPECT MUCH HERE ({len(none)} combos):")
    for _, r in none.iterrows():
        print(f"  {r.resource:<10} h={int(r.horizon):<3} VR p={r.vr_p:.4f}  Hurst={r.hurst:.3f}")

    print("\nwrote stage0_diagnostics.csv + stage0_series_stats.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
