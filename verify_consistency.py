"""H21 verification harness (spec section 5).

Checks, for every trained artifact:
  1. leakage assertions still pass per resource (scramble test + static pattern scan)
  2. row-count / null-count sanity per resource dataset
  3. the consolidated leaderboard, the per-model metadata, and the values the Streamlit app
     reads are IDENTICAL - no silent recomputation drift between training and the app
  4. artifact completeness (every resource x horizon present and loadable)

Exit code 0 = all checks pass. Non-zero = something to fix before trusting the app.

Usage:  python verify_consistency.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

import h21_core as C

FAILURES: list[str] = []
CHECKS = 0


def check(cond: bool, label: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        FAILURES.append(f"{label} {detail}")


def main() -> int:
    print("=" * 84)
    print("H21 CONSISTENCY & LEAKAGE VERIFICATION".center(84))
    print("=" * 84)

    d = C.load_data()

    # ---------------------------------------------------------------- 1. leakage
    print("\n[1] LEAKAGE ASSERTIONS (per resource)")
    try:
        C.assert_no_forbidden_patterns()
        check(True, "static scan: no bfill / center=True / negative shift in feature code")
    except AssertionError as e:
        check(False, "static scan", str(e))

    for res in C.RESOURCES:
        try:
            C.assert_causal(d, res)
            check(True, f"causality scramble test [{res}]")
        except AssertionError as e:
            check(False, f"causality scramble test [{res}]", str(e)[:160])

    # ---------------------------------------------------------------- 2. dataset sanity
    print("\n[2] ROW-COUNT / NULL SANITY (per resource)")
    for res in C.RESOURCES:
        X, Y, P, D = C.build_dataset(d, res)
        check(len(X) == len(Y) == len(P) == len(D),
              f"aligned lengths [{res}]", f"X={len(X)} Y={len(Y)} P={len(P)} D={len(D)}")
        check(P.notna().all(), f"no missing target rows survived [{res}]",
              f"{int(P.isna().sum())} nulls")
        check(D.is_monotonic_increasing, f"chronological order [{res}]")
        # every column must be either fully usable or explainably sparse, never all-null
        allnull = [c for c in X.columns if X[c].isna().all()]
        check(not allnull, f"no all-null feature columns [{res}]", f"{allnull[:5]}")

    # ---------------------------------------------------------------- 3. artifacts
    print("\n[3] ARTIFACT COMPLETENESS")
    arts = sorted(glob.glob(os.path.join(C.MODEL_DIR, "*.joblib")))
    if not arts:
        print("  (no artifacts yet - run `python train_all.py` first)")
        return 1
    check(len(arts) > 0, f"artifacts found", f"n={len(arts)}")

    loaded = {}
    for path in arts:
        name = os.path.basename(path)[: -len(".joblib")]
        meta_path = os.path.join(C.MODEL_DIR, f"{name}.meta.json")
        ok = os.path.exists(meta_path)
        check(ok, f"metadata sidecar exists [{name}]")
        if not ok:
            continue
        try:
            art = joblib.load(path)
            meta = json.load(open(meta_path, encoding="utf-8"))
            loaded[name] = (art, meta)
        except Exception as e:
            check(False, f"loadable [{name}]", str(e)[:120])

    # ---------------------------------------------------------------- 4. no drift
    print("\n[4] LEADERBOARD <-> METADATA <-> APP CONSISTENCY")
    lb_path = "leaderboard.csv"
    if not os.path.exists(lb_path):
        check(False, "leaderboard.csv exists")
        return 1
    LB = pd.read_csv(lb_path)
    check(len(LB) == len(loaded),
          "leaderboard row count == artifact count", f"lb={len(LB)} artifacts={len(loaded)}")

    FIELDS = [("test_rmse", "test_rmse"), ("test_mae", "test_mae"), ("test_r2", "test_r2"),
              ("test_diracc", "test_diracc"), ("test_coverage", "test_coverage"),
              ("test_n_dir", "test_n_dir"), ("naive_rmse", "naive_rmse"),
              ("dm_p", "dm_p"), ("beats_naive", "beats_naive")]

    drift = 0
    for _, row in LB.iterrows():
        name = f"{row['resource']}_h{int(row['horizon'])}"
        if name not in loaded:
            check(False, f"leaderboard row has an artifact [{name}]")
            continue
        art, meta = loaded[name]
        for lb_col, meta_key in FIELDS:
            lv, mv = row[lb_col], meta["metrics"][meta_key]
            # the app reads meta["metrics"]; training wrote both from the same dict
            av = art["metrics"][meta_key]
            same = True
            for a, b in ((lv, mv), (mv, av)):
                if isinstance(b, bool) or isinstance(a, (bool, np.bool_)):
                    same &= bool(a) == bool(b)
                elif a is None or b is None or (isinstance(a, float) and np.isnan(a)):
                    same &= (a is None or (isinstance(a, float) and np.isnan(a))) == \
                            (b is None or (isinstance(b, float) and np.isnan(b)))
                else:
                    same &= abs(float(a) - float(b)) < 1e-9
            if not same:
                drift += 1
                print(f"  FAIL  drift [{name}.{lb_col}] leaderboard={lv} meta={mv} artifact={av}")
    check(drift == 0, "zero drift between leaderboard, metadata and artifact metrics",
          f"{drift} mismatched fields")

    # ---------------------------------------------------------------- 5. honesty flags
    print("\n[5] HONESTY FLAGS (reported, not failures)")
    if "beats_naive" in LB.columns:
        bad = LB[~LB.beats_naive.astype(bool)]
        print(f"  {len(bad)} / {len(LB)} resource-horizon combos DO NOT beat the naive baseline")
        for _, r in bad.iterrows():
            print(f"    - {r['resource']:<10} h={int(r['horizon']):<3} "
                  f"RMSE={r['test_rmse']:.2f} vs naive {r['naive_rmse']:.2f}")
        sig = LB[(LB.beats_naive.astype(bool)) & (LB.dm_p < 0.05)]
        print(f"  {len(sig)} / {len(LB)} beat naive with Diebold-Mariano p < 0.05")

    print("\n" + "=" * 84)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED / {CHECKS} checks")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"RESULT: ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
