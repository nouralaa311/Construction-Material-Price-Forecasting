"""Verification for the three RedCon changes (spec section 4).

Exercises the REAL app functions (imported from redconapp) with Streamlit's caching stubbed
out, so this proves the app's own code path behaves - not a reimplementation of it.

Run:  python test_app_paths.py
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

# ---- stub streamlit so redconapp is importable headless --------------------------------
st_stub = types.ModuleType("streamlit")


def _identity_decorator(*a, **k):
    def deco(fn):
        return fn
    if a and callable(a[0]):
        return a[0]
    return deco


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(st_stub, name, lambda *a, **k: None)


def _cols(spec, *a, **k):
    n = spec if isinstance(spec, int) else len(spec)
    return [_Ctx() for _ in range(n)]


for name in ("title", "subheader", "markdown", "caption", "write", "metric", "divider", "error",
             "warning", "info", "success", "dataframe", "json", "plotly_chart", "button",
             "stop", "rerun", "set_page_config", "header"):
    setattr(st_stub, name, lambda *a, **k: None)


# Widgets return a REAL selection rather than None, so the app's top-level body actually runs
# end-to-end. With None-returning stubs the body died immediately and every real crash after
# that point was invisible to this test.
def _selectbox(label, options=None, index=0, **k):
    opts = list(options or [])
    return opts[index] if opts and index is not None and index < len(opts) else (opts[0] if opts else None)


def _multiselect(label, options=None, default=None, **k):
    if default is not None:
        return list(default)
    return list(options or [])[:2]


st_stub.selectbox = _selectbox
st_stub.multiselect = _multiselect
st_stub.checkbox = lambda label, value=False, **k: value
st_stub.slider = lambda label, mn=None, mx=None, value=None, **k: value if value is not None else mn
st_stub.number_input = lambda label, min_value=None, max_value=None, value=None, **k: (
    value if value is not None else (min_value or 0))
st_stub.date_input = lambda label, value=None, **k: value
st_stub.warning = lambda *a, **k: None
st_stub.columns = _cols
st_stub.expander = lambda *a, **k: _Ctx()
st_stub.sidebar = _Ctx()
st_stub.cache_data = _identity_decorator
st_stub.cache_data.clear = lambda: None
st_stub.cache_resource = _identity_decorator
st_stub.cache_resource.clear = lambda: None
st_stub.session_state = {}
sys.modules["streamlit"] = st_stub

import h21_core as C  # noqa: E402

# import the app's functions without executing its UI body
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("_redcon_fns", "redconapp.py")
mod = importlib.util.module_from_spec(spec)
APP_BODY_ERROR = None
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass
except Exception as e:
    # A crash here means the app's TOP-LEVEL body is broken and the real app would render a red
    # error page. This used to be printed and ignored, which let a genuine KeyError ship.
    import traceback
    APP_BODY_ERROR = f"{type(e).__name__}: {e}"
    print(f"\n*** APP BODY CRASHED: {APP_BODY_ERROR}")
    traceback.print_exc()

FAIL = []


def pick(models, r, h):
    """models[r][h] is now {family: paths}; pick the best-by-walk-forward family."""
    fams = models[r][h]
    import math
    def wf(f):
        try:
            import json as _j
            return _j.load(open(fams[f]["meta"], encoding="utf-8"))["metrics"].get(
                "val_rmse", math.inf)
        except Exception:
            return math.inf
    return fams[min(fams, key=wf)]


def check(cond, label, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + ("" if cond else f"   {detail}"))
    if not cond:
        FAIL.append(label)


def main():
    print("=" * 78)
    print("REDCON APP VERIFICATION".center(78))
    print("=" * 78)

    models, _sk = mod.usable_models(mod.models_signature())
    resources = sorted(models)
    print(f"\ndiscovered resources: {resources}")
    d = C.load_data()

    # ---------------------------------------------------------------- 1. resource switching
    print("\n[1] RESOURCE SELECTOR ACTUALLY CHANGES OUTPUT")
    check(len(resources) >= 2, "at least two resources available to compare",
          f"found {len(resources)}")
    results = {}
    for r in resources:
        h = 21 if 21 in models[r] else max(models[r])
        art = mod.artifact_for(pick(models, r, h))
        meta = mod.meta_for(pick(models, r, h))
        p = mod.predict_latest(art, d)
        results[r] = dict(pred=p["pred_price"], last=p["last_price"],
                          rmse=meta["metrics"]["test_rmse"], res_in_art=art["resource"],
                          col=C.RESOURCES[r]["column"])
        check(art["resource"] == r, f"artifact identity matches selection [{r}]",
              f"artifact says {art['resource']}")
        # the anchor price must equal that resource's own last real quote
        own_last = float(d[C.RESOURCES[r]["column"]].dropna().iloc[-1])
        check(abs(p["last_price"] - own_last) < 1e-6,
              f"anchor price is {r}'s own series", f"{p['last_price']} vs {own_last}")

    preds = [round(v["pred"], 6) for v in results.values()]
    check(len(set(preds)) == len(preds), "every resource yields a DISTINCT forecast",
          f"{preds}")
    rmses = [round(v["rmse"], 6) for v in results.values()]
    check(len(set(rmses)) == len(rmses), "every resource yields DISTINCT saved metrics", f"{rmses}")
    for r, v in results.items():
        print(f"      {r:<10} anchor={v['last']:>12,.2f}  forecast={v['pred']:>12,.2f}  "
              f"test_rmse={v['rmse']:,.2f}")

    # ---------------------------------------------------------------- 2. live price input
    print("\n[2] MANUAL 'AS-OF TODAY' PRICE RECOMPUTES FEATURES")
    r = resources[0]
    col = C.RESOURCES[r]["column"]
    X0, _, P0, D0 = C.build_dataset(d, r)
    before = {f: float(X0[f].iloc[-1]) for f in ("own_lag_1", "own_mean_5", "own_ret_1")}
    prev_actual = float(P0.iloc[-1])

    bumped = prev_actual * 1.10
    newdate = pd.Timestamp(D0.iloc[-1]) + pd.Timedelta(days=1)
    d_live, notes = C.append_live_row(d, r, newdate, bumped)
    X1, _, P1, _ = C.build_dataset(d_live, r)
    after = {f: float(X1[f].iloc[-1]) for f in ("own_lag_1", "own_mean_5", "own_ret_1")}

    print(f"      resource={r}  {prev_actual:,.2f} -> manual {bumped:,.2f}")
    for f in before:
        print(f"      {f:<12} {before[f]:>14,.4f}  ->  {after[f]:>14,.4f}")

    check(abs(float(P1.iloc[-1]) - bumped) < 1e-6, "new anchor price is the entered value")
    check(abs(after["own_lag_1"] - prev_actual) < 1e-6,
          "own_lag_1 == previous real price (lag correctly shifted)",
          f"{after['own_lag_1']} vs {prev_actual}")
    check(abs(after["own_ret_1"] - (bumped / prev_actual - 1)) < 1e-6,
          "own_ret_1 recomputed from the entered value")
    check(abs(after["own_mean_5"] - before["own_mean_5"]) > 1e-9,
          "rolling mean actually changed (not stale)")
    check(len(X1) == len(X0) + 1, "exactly one row added", f"{len(X0)} -> {len(X1)}")
    check(d.shape == C.load_data().shape, "source data NOT mutated (inference-only)")

    ext = C.external_features_asof(newdate)
    check(len(ext) > 0, "external features recomputed as-of the live date", f"n={len(ext)}")

    live_pred = mod.predict_latest(
        mod.artifact_for(pick(models, r, 21 if 21 in models[r] else max(models[r]))), d_live)
    check(abs(live_pred["pred_price"] - results[r]["pred"]) > 1e-9,
          "live forecast differs from stored-history forecast",
          f"{live_pred['pred_price']} vs {results[r]['pred']}")

    # ---------------------------------------------------------------- 3. free-form horizon
    print("\n[3] FREE-FORM HORIZON: INTERPOLATION vs EXTRAPOLATION")
    hs = sorted(models[r])
    p_all = {h: mod.predict_latest(mod.artifact_for(pick(models, r, h)), d) for h in hs}
    last_price = p_all[hs[0]]["last_price"]
    max_day = max(C.HORIZON_DAYS[h] for h in hs)

    inside = mod.freeform_forecast(p_all, 10.0, last_price)
    check(inside is not None and inside["mode"] == "interpolated",
          "10 days (between h=7 and h=14) -> interpolated",
          f"{inside['mode'] if inside else None}")
    check(inside["nearest_h"] in hs, "reports a real nearest trained horizon",
          f"{inside['nearest_h']}")
    print(f"      10d -> {inside['value']:,.2f} ({inside['mode']}, "
          f"between {inside['bracket'][0]} and {inside['bracket'][1]}, "
          f"nearest trained h={inside['nearest_h']})")

    outside = mod.freeform_forecast(p_all, 90.0, last_price)
    check(outside is not None and outside["mode"] == "extrapolated",
          f"90 days (beyond {max_day:.1f}d) -> extrapolated",
          f"{outside['mode'] if outside else None}")
    check(outside["nearest_h"] == max(hs), "nearest trained horizon for a far date is the max h",
          f"{outside['nearest_h']}")
    print(f"      90d -> {outside['value']:,.2f} ({outside['mode']}, "
          f"nearest trained h={outside['nearest_h']})")

    exact = mod.freeform_forecast(p_all, C.HORIZON_DAYS[hs[0]], last_price)
    check(abs(exact["value"] - p_all[hs[0]]["pred_price"]) < 1e-6,
          f"requesting exactly h={hs[0]}'s day count reproduces its own prediction",
          f"{exact['value']} vs {p_all[hs[0]]['pred_price']}")

    # ---------------------------------------------------------------- 4. family selector
    print("\n[4] MODEL FAMILY SELECTOR CHANGES OUTPUT")
    r2 = h2 = None
    for cand in resources:
        hh = 21 if 21 in models[cand] else max(models[cand])
        if len(models[cand][hh]) >= 2:
            r2, h2 = cand, hh
            break
    if r2 is None:
        check(False, "some resource has >=2 trained families", "none found yet")
    else:
        fams = sorted(models[r2][h2])
        print(f"      {r2} h={h2}: {len(fams)} families -> {fams}")
        seen = {}
        for f in fams:
            art = mod.artifact_for(models[r2][h2][f])
            mt = mod.meta_for(models[r2][h2][f])["metrics"]
            pr = mod.predict_latest(art, d)
            seen[f] = (round(pr["pred_price"], 6), round(mt["test_rmse"], 6))
        vals = [v[0] for v in seen.values()]
        rms = [v[1] for v in seen.values()]
        check(len(set(vals)) > 1, "different families give DIFFERENT forecasts", f"{vals}")
        check(len(set(rms)) > 1, "different families give DIFFERENT saved metrics", f"{rms}")
        for f, (pv, rm) in seen.items():
            print(f"      {f:<28} forecast={pv:>12,.2f}  test_rmse={rm:>10,.2f}")
        pm = mod.persistence_metrics(r2, h2, d)
        check(bool(pm), "persistence baseline computes")
        if pm:
            print(f"      {'PERSISTENCE (naive)':<28} {'':>12}  test_rmse={pm['test_rmse']:>10,.2f}")

    print("\n" + "=" * 78)
    check(APP_BODY_ERROR is None, "app top-level body executes without crashing",
          APP_BODY_ERROR or "")

    if FAIL:
        print(f"RESULT: {len(FAIL)} FAILED")
        for f in FAIL:
            print("  - " + f)
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
