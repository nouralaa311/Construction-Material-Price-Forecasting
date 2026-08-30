"""RedCon — H21 Multi-Resource Construction Price Forecasting.

Resource-agnostic: the sidebar is populated by scanning `models/`, so adding a resource is a
training-run change, never a UI-code change.

Metric integrity: historical/backtest metrics are READ from the artifacts training wrote. The app
computes no metric of its own, which is what guarantees these match `leaderboard.csv`.

Two clearly separated number types, never mixed in the UI:
  * "Historical walk-forward performance" - frozen, from the saved artifacts.
  * "Live forecast from your input"       - inference-only, computed now from a price you type.

Run:  python -m streamlit run redconapp.py
"""
from __future__ import annotations

import glob
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import h21_core as C

st.set_page_config(page_title="RedCon — Resource Forecasting", page_icon="🏗️", layout="wide")


# =======================================================================================
# Artifact discovery
# =======================================================================================
def models_signature() -> tuple:
    """Fingerprint of the models directory. Passed into the cached discovery function so that
    newly-trained artifacts appear WITHOUT a restart. (The original bug: discovery was cached
    with no key, so a session started mid-training kept showing only the resources that
    existed at first load.)"""
    out = []
    for p in sorted(glob.glob(os.path.join(C.MODEL_DIR, "*.joblib"))):
        try:
            out.append((os.path.basename(p), int(os.path.getmtime(p))))
        except OSError:
            pass
    return tuple(out)


@st.cache_data(show_spinner=False)
def discover_models(signature: tuple) -> dict:
    """{resource: {horizon: {family: paths}}}.

    Two filename shapes are recognised:
        {res}_h{h}__{family}.joblib   -> a specific model family
        {res}_h{h}.joblib             -> the campaign's selected winner (family read from meta)
    Families are discovered from what EXISTS on disk, so a family that was never trained for a
    resource simply does not appear - it is never silently substituted with another model.
    """
    found: dict = {}
    for path in sorted(glob.glob(os.path.join(C.MODEL_DIR, "*.joblib"))):
        name = os.path.basename(path)[: -len(".joblib")]
        meta_path = os.path.join(C.MODEL_DIR, f"{name}.meta.json")
        if not os.path.exists(meta_path):
            continue
        m = re.fullmatch(r"(.+?)_h(\d+)__(.+)", name)
        if m:
            res, h, fam = m.group(1), int(m.group(2)), m.group(3)
        else:
            m = re.fullmatch(r"(.+?)_h(\d+)", name)
            if not m:
                continue
            res, h = m.group(1), int(m.group(2))
            try:
                fam = json.load(open(meta_path, encoding="utf-8"))["meta"].get("family", "selected")
            except Exception:
                fam = "selected"
            fam = f"{fam} (campaign pick)"
        found.setdefault(res, {}).setdefault(h, {})[fam] = dict(artifact=path, meta=meta_path)
    return found


@st.cache_data(show_spinner=False)
def usable_models(signature: tuple) -> tuple:
    """Discovery, minus any artifact that cannot actually be used for inference.

    A deep (GRU/BiGRU) artifact needs its fitted scalers; a tabular one needs a .predict.
    An artifact failing either check is EXCLUDED from the selector rather than offered and
    then failing at predict time - but it is also counted and reported, never hidden.
    """
    raw = discover_models(signature)
    good, skipped = {}, []
    for res, by_h in raw.items():
        for h, fams in by_h.items():
            for fam, paths in fams.items():
                try:
                    art = joblib.load(paths["artifact"])
                    ok = hasattr(art["model"], "predict") or (
                        "feature_scaler" in art and "target_scaler" in art)
                except Exception as e:
                    ok, art = False, None
                    skipped.append((res, h, fam, f"unreadable: {type(e).__name__}"))
                if ok:
                    good.setdefault(res, {}).setdefault(h, {})[fam] = paths
                elif art is not None:
                    skipped.append((res, h, fam, "deep artifact missing fitted scalers"))
    return good, tuple(skipped)


@st.cache_data(show_spinner=False)
def load_meta(meta_path: str, mtime: float) -> dict:
    with open(meta_path, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_resource(show_spinner=False)
def load_artifact(path: str, mtime: float):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_prices(mtime: float) -> pd.DataFrame:
    return C.load_data()


def meta_for(paths: dict) -> dict:
    return load_meta(paths["meta"], os.path.getmtime(paths["meta"]))


def artifact_for(paths: dict):
    return load_artifact(paths["artifact"], os.path.getmtime(paths["artifact"]))


# =======================================================================================
# Inference  (never touches training data or saved artifacts)
# =======================================================================================
def predict_latest(artifact: dict, d: pd.DataFrame):
    """Forecast h steps beyond the last row of `d`.

    `d` is passed in explicitly so the same function serves both the stored history and a
    live what-if frame carrying a manually entered price. Inference only.
    """
    resource, h, kind = artifact["resource"], artifact["horizon"], artifact["target_kind"]
    X, Y, P, D = C.build_dataset(d, resource)
    if len(X) == 0:
        return None

    # Deep families store a torch state_dict; tabular families store an object with .predict.
    # Dispatch on the OBJECT, not on the family string - checking `family != "gru"` sent
    # `bigru` down the tabular path and crashed on state_dict.predict().
    if hasattr(artifact["model"], "predict"):
        feats = [f for f in artifact["features"] if f in X.columns]
        Xi, _ = C.impute_fold(X[feats], [], artifact["strategy"])
        raw = artifact["model"].predict(Xi.iloc[[-1]])[0]
    else:
        from train_all import GRUForecaster, make_sequences, gru_predict
        gfeats = [f for f in artifact["gru_features"] if f in X.columns]
        _, (Xfull,) = C.impute_fold(X[gfeats], [X[gfeats]], "ffill_median")
        Xm = np.clip(np.nan_to_num(
            artifact["feature_scaler"].transform(Xfull.values).astype(np.float32)), -10, 10)
        p = artifact["params"]
        model = GRUForecaster(Xm.shape[1], int(p["hidden"]), int(p["layers"]),
                              float(p["dropout"]), p.get("arch", artifact.get("family", "gru")))
        model.load_state_dict(artifact["model"])
        model.eval()
        L = int(p["seq_len"])
        raw = artifact["target_scaler"].inverse_transform(
            gru_predict(model, Xm[-L:][None, ...].astype(np.float32)).reshape(-1, 1)).ravel()[0]

    last_price = float(P.iloc[-1])
    last_date = D.iloc[-1]
    pred_price = float(C.target_to_price([raw], [last_price], kind)[0])
    return dict(last_price=last_price, last_date=last_date, pred_price=pred_price,
                pct_change=(pred_price / last_price - 1.0) * 100.0,
                target_date=last_date + pd.Timedelta(days=C.HORIZON_DAYS[h]),
                history_dates=D, history_prices=P)


@st.cache_data(show_spinner=False)
def predict_history(_artifact: dict, cache_key: str, _d: pd.DataFrame) -> pd.DataFrame:
    """Model track record 'in arrears': at date t, what the model predicted for t, h steps
    earlier. Rows before test_start are in-sample fit; only after it is out-of-sample."""
    artifact, d = _artifact, _d
    resource, h, kind = artifact["resource"], artifact["horizon"], artifact["target_kind"]
    X, Y, P, D = C.build_dataset(d, resource)
    n = len(X)
    if n <= h:
        return pd.DataFrame(columns=["date", "model"])

    if hasattr(artifact["model"], "predict"):
        feats = [f for f in artifact["features"] if f in X.columns]
        Xi, _ = C.impute_fold(X[feats], [], artifact["strategy"])
        raw = artifact["model"].predict(Xi)
        src = np.arange(n)
    else:
        from train_all import GRUForecaster, make_sequences, gru_predict
        gfeats = [f for f in artifact["gru_features"] if f in X.columns]
        _, (Xfull,) = C.impute_fold(X[gfeats], [X[gfeats]], "ffill_median")
        Xm = np.clip(np.nan_to_num(
            artifact["feature_scaler"].transform(Xfull.values).astype(np.float32)), -10, 10)
        p = artifact["params"]
        model = GRUForecaster(Xm.shape[1], int(p["hidden"]), int(p["layers"]),
                              float(p["dropout"]), p.get("arch", artifact.get("family", "gru")))
        model.load_state_dict(artifact["model"])
        model.eval()
        S, _, keep = make_sequences(Xm, np.zeros(n), int(p["seq_len"]), np.arange(n))
        raw = artifact["target_scaler"].inverse_transform(
            gru_predict(model, S).reshape(-1, 1)).ravel()
        src = keep

    pred_price = C.target_to_price(raw, P.values[src], kind)
    tgt = src + h
    ok = tgt < n
    return pd.DataFrame({"date": D.iloc[tgt[ok]].values, "model": pred_price[ok]})


def persistence_metrics(resource: str, h: int, d: pd.DataFrame) -> dict:
    """The naive/persistence baseline computed on the SAME test window the models use, so the
    comparison row in the table is apples-to-apples."""
    X, Y, P, D = C.build_dataset(d, resource)
    emb = max(C.HORIZONS)
    _, _, te = C.make_splits(len(X), emb)
    yt = Y[f"y_price_h{h}"].values[te]
    b = P.values[te]
    ok = ~np.isnan(yt)
    if ok.sum() < 5:
        return {}
    ev = C.evaluate(yt[ok], b[ok], b[ok], 0.0, "persistence", h)
    return dict(test_rmse=ev["RMSE"], test_mae=ev["MAE"], test_r2=ev["R2"],
                test_diracc=ev["DirAcc"], test_coverage=ev["Coverage"],
                test_n_dir=ev["N_dir"], naive_rmse=ev["RMSE"], beats_naive=False,
                val_rmse=float("nan"))


def signal_of(pct_change: float, dead_zone: float) -> tuple:
    dz = dead_zone * 100.0
    if abs(pct_change) < dz:
        return "FLAT", "#6b7280", "➖"
    return ("UP", "#059669", "▲") if pct_change > 0 else ("DOWN", "#dc2626", "▼")


def freeform_forecast(preds: dict, req_days: float, last_price: float):
    """Answer an arbitrary horizon from the 5 trained ones.

    Models exist ONLY at h=1,3,7,14,21. Anything else is interpolated between the two nearest
    trained horizons, or extrapolated beyond the last one - and is labelled as such. The
    anchor at 0 days is the current price, which is a real known quantity.
    """
    avail = {h: p for h, p in preds.items() if p}
    if not avail:
        return None
    hs = sorted(avail, key=lambda h: C.HORIZON_DAYS[h])
    xs = [0.0] + [C.HORIZON_DAYS[h] for h in hs]
    ys = [last_price] + [avail[h]["pred_price"] for h in hs]
    max_day = xs[-1]

    nearest_h = min(hs, key=lambda h: abs(C.HORIZON_DAYS[h] - req_days))

    if req_days <= max_day:
        value = float(np.interp(req_days, xs, ys))
        lo = max([i for i in range(len(xs)) if xs[i] <= req_days], default=0)
        hi = min(lo + 1, len(xs) - 1)
        bracket = (("current price" if lo == 0 else f"h={hs[lo-1]}"), f"h={hs[hi-1]}" if hi > 0 else "")
        mode = "interpolated"
    else:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if xs[-1] != xs[-2] else 0.0
        value = float(ys[-1] + slope * (req_days - max_day))
        bracket = (f"h={hs[-2]}", f"h={hs[-1]}")
        mode = "extrapolated"

    return dict(value=value, mode=mode, bracket=bracket, nearest_h=nearest_h,
                max_day=max_day, pct=(value / last_price - 1.0) * 100.0)


# =======================================================================================
# UI
# =======================================================================================
models, _skipped = usable_models(models_signature())

st.title("🏗️ RedCon — Construction Resource Price Forecasting")

if not models:
    st.error(f"No trained models found in `{C.MODEL_DIR}/`.\n\nTrain them:\n```\npython train_all.py\n```")
    st.stop()

# ---- sidebar: resource ---------------------------------------------------------------
st.sidebar.header("Resource")
def _any_paths(res: str) -> dict:
    """models[res] is {horizon: {family: paths}} - descend BOTH levels to reach a paths dict."""
    by_h = models[res]
    fams = next(iter(by_h.values()))
    return next(iter(fams.values()))


labels = {r: meta_for(_any_paths(r))["meta"].get("label", r.title()) for r in sorted(models)}
choice = st.sidebar.selectbox("Select a resource", options=sorted(models),
                              format_func=lambda r: labels.get(r, r.title()),
                              key="resource_choice")
_n_art = sum(len(f) for by_h in models.values() for f in by_h.values())
st.sidebar.caption(f"{len(models)} resource(s) · {_n_art} usable model artifacts")
if _skipped:
    with st.sidebar.expander(f"{len(_skipped)} artifact(s) excluded as unusable"):
        for r_, h_, f_, why in _skipped:
            st.write(f"- {r_} h={h_} {f_}: {why}")
        st.caption("Excluded from the selector so they cannot silently fail at predict time. "
                   "Re-run training for these combinations.")
if st.sidebar.button("🔄 Refresh model list", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

horizons = sorted(models[choice])

# ---- horizon selector --------------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("Horizon")
_default_h = 21 if 21 in horizons else max(horizons)
headline_h = st.sidebar.selectbox(
    "Forecast horizon", options=horizons,
    index=horizons.index(_default_h),
    format_func=lambda x: f"h={x}  (≈{C.HORIZON_DAYS[x]} cal. days)",
    key=f"horizon_{choice}",
    help="h is a step count (observations), not calendar days. Each horizon has its own "
         "independently trained models.")

# ---- model family selector (populated ONLY from artifacts that exist) --------------
st.sidebar.divider()
st.sidebar.header("Model")
fams_at_headline = sorted(models[choice].get(headline_h, {}))
if not fams_at_headline:
    st.error(f"No trained model for {choice} at h={headline_h}.")
    st.stop()


def _fam_label(f: str) -> str:
    try:
        mm = meta_for(models[choice][headline_h][f])["metrics"]
        wf = mm.get("val_rmse")
        sk = mm.get("wf_skill")
        extra = f" — WF {wf:,.1f}" if wf and np.isfinite(wf) else ""
        extra += f" (skill {sk:.2f})" if sk and np.isfinite(sk) else ""
        return f + extra
    except Exception:
        return f


default_fam = min(
    fams_at_headline,
    key=lambda f: meta_for(models[choice][headline_h][f])["metrics"].get("val_rmse", float("inf")))
family = st.sidebar.selectbox(
    "Model family", options=fams_at_headline,
    index=fams_at_headline.index(default_fam), format_func=_fam_label,
    key=f"family_{choice}",
    help="Only families actually trained for this resource are listed. The number is the "
         "walk-forward score — lower is better; skill <1 beats persistence.")
st.sidebar.caption(f"{len(fams_at_headline)} families trained for {labels[choice]} at h={headline_h}")

# a family may not exist at every horizon; fall back per-horizon ONLY with a visible note
arts, metas, fam_used = {}, {}, {}
for h in horizons:
    avail = models[choice][h]
    use = family if family in avail else (default_fam if default_fam in avail else sorted(avail)[0])
    fam_used[h] = use
    arts[h] = artifact_for(avail[use])
    metas[h] = meta_for(avail[use])
missing = [h for h in horizons if fam_used[h] != family]
if missing:
    st.sidebar.warning(f"'{family}' was not trained at h={missing} — those cards use "
                       f"'{fam_used[missing[0]]}' instead (shown explicitly, never silently).")
unit = metas[headline_h]["meta"].get("unit", "")

d_hist = load_prices(os.path.getmtime(C.DATA_PATH))

# ---- sidebar: live price input -------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("Live price input")
st.sidebar.caption(f"Optional. Enter today's **{labels[choice]}** price to run a live forecast "
                   "from it. Inference only — nothing is retrained or written to disk.")

_Xd, _Yd, _Pd, _Dd = C.build_dataset(d_hist, choice)
stored_last_date, stored_last_price = _Dd.iloc[-1], float(_Pd.iloc[-1])

use_live = st.sidebar.checkbox("Use my own current price", value=False, key=f"uselive_{choice}")
live_date = st.sidebar.date_input("As-of date", value=pd.Timestamp.today().date(),
                                  key=f"livedate_{choice}", disabled=not use_live)
live_price = st.sidebar.number_input(f"Price ({unit})", min_value=0.0, value=stored_last_price,
                                     step=max(0.01, round(stored_last_price * 0.001, 2)),
                                     format="%.2f", key=f"liveprice_{choice}",
                                     disabled=not use_live)

live_notes: list[str] = []
if use_live and live_price > 0:
    d_active, live_notes = C.append_live_row(d_hist, choice, pd.Timestamp(live_date), live_price)
    mode_label = "Live forecast from your input"
else:
    d_active = d_hist
    mode_label = "Forecast from stored history"

# ---- predictions ---------------------------------------------------------------------
preds = {}
for h in horizons:
    try:
        preds[h] = predict_latest(arts[h], d_active)
    except Exception as e:
        st.warning(f"h={h}: prediction unavailable ({e})")
        preds[h] = None

live = next((p for p in preds.values() if p), None)
if live is None:
    st.error("No horizon produced a usable prediction for this resource.")
    st.stop()

st.subheader(f"{labels[choice]} · **{family}** · headline horizon **h={headline_h}**")
if use_live:
    st.info(f"**{mode_label}** — anchored on your entered price "
            f"**{live_price:,.2f} {unit}** as of **{live_date}**. "
            "All lag/rolling/derived features were rebuilt from (real history + your value), and "
            "external drivers were recomputed as-of that date using the pipeline's "
            "publication-lag rules. This is inference only — training data and saved models are "
            "untouched.")
    with st.expander("What changed for this live run"):
        for n in live_notes:
            st.write("• " + n)
        st.write(f"• Stored last observation: **{stored_last_price:,.2f}** on "
                 f"{stored_last_date.date()} → live anchor: **{live_price:,.2f}** on {live_date}.")

c1, c2, c3 = st.columns(3)
c1.metric("Anchor price", f"{live['last_price']:,.2f}", help=unit)
c2.metric("As of", pd.Timestamp(live["last_date"]).strftime("%Y-%m-%d"))
hp = preds.get(headline_h)
if hp:
    c3.metric(f"Forecast (h={headline_h})", f"{hp['pred_price']:,.2f}",
              delta=f"{hp['pct_change']:+.2f}%")

st.divider()

# ---- per-horizon cards ---------------------------------------------------------------
st.markdown("### Forecast by trained horizon")
st.caption("`h` is a step count (observations), not calendar days — sampling is irregular. "
           "Track-record figures come from the saved artifacts (historical walk-forward), and are "
           "unaffected by any live price you enter.")

cols = st.columns(len(horizons))
for col, h in zip(cols, horizons):
    p, mt = preds[h], metas[h]["metrics"]
    with col:
        st.markdown(f"**h = {h}**  \n<span style='color:#6b7280;font-size:0.8em'>"
                    f"≈ {C.HORIZON_DAYS[h]} cal. days</span>", unsafe_allow_html=True)
        if not p:
            st.write("—")
            continue
        sig, colr, arrow = signal_of(p["pct_change"], metas[h]["meta"]["dead_zone"])
        st.markdown(f"<div style='font-size:1.35em;font-weight:700'>{p['pred_price']:,.2f}</div>"
                    f"<div style='color:{colr};font-weight:600'>{arrow} {sig} "
                    f"{p['pct_change']:+.2f}%</div>"
                    f"<div style='color:#6b7280;font-size:0.8em'>"
                    f"{pd.Timestamp(p['target_date']).strftime('%Y-%m-%d')}</div>",
                    unsafe_allow_html=True)
        da, cov = mt.get("test_diracc"), mt.get("test_coverage")
        if da is not None and not pd.isna(da):
            st.caption(f"track record: **{da:.1%}** dir. acc  \n({cov:.0%} coverage, "
                       f"n={mt.get('test_n_dir')}, p={mt.get('test_p_value'):.3f})")
        else:
            st.caption("track record: n/a")
        if not mt.get("beats_naive", False):
            st.markdown("<span style='color:#dc2626;font-size:0.78em'>⚠ does not beat naive</span>",
                        unsafe_allow_html=True)

# ---- free-form horizon ---------------------------------------------------------------
st.divider()
st.markdown("### Custom horizon")
max_day = max(C.HORIZON_DAYS[h] for h in horizons)
f1, f2 = st.columns([1, 3])
req_days = f1.number_input("Days from anchor", min_value=1, max_value=365, value=int(round(max_day)),
                           step=1, key=f"req_{choice}")
ff = freeform_forecast(preds, float(req_days), live["last_price"])

if ff:
    target_date = pd.Timestamp(live["last_date"]) + pd.Timedelta(days=float(req_days))
    with f2:
        if ff["mode"] == "extrapolated":
            st.error(f"**Extrapolated — low confidence.** No model was trained or validated "
                     f"beyond ≈{max_day:.1f} calendar days (h={max(horizons)}). You asked for "
                     f"{req_days} days, which is **{req_days - max_day:.1f} days past** the "
                     "furthest trained horizon. This number is a straight-line projection off the "
                     "end of the trained range — treat it as an illustration, not a forecast.")
        else:
            st.warning(f"**Interpolated estimate** between {ff['bracket'][0]} and "
                       f"{ff['bracket'][1]} — no model is trained at exactly {req_days} days. "
                       "Linear interpolation on calendar days.")
        g1, g2 = st.columns(2)
        g1.metric(f"{ff['mode'].title()} ({req_days}d → {target_date.date()})",
                  f"{ff['value']:,.2f}", delta=f"{ff['pct']:+.2f}%")
        nh = ff["nearest_h"]
        np_ = preds[nh]
        nm = metas[nh]["metrics"]
        g2.metric(f"Nearest TRAINED horizon — h={nh} (≈{C.HORIZON_DAYS[nh]}d)",
                  f"{np_['pred_price']:,.2f}", delta=f"{np_['pct_change']:+.2f}%")
        da = nm.get("test_diracc")
        st.caption(
            f"**Grounded comparison — h={nh}** is directly trained and validated: "
            f"walk-forward RMSE {nm['val_rmse']:,.2f} · test RMSE {nm['test_rmse']:,.2f} · "
            f"R² {nm['test_r2']:.3f} · directional accuracy "
            f"{'n/a' if da is None or pd.isna(da) else f'{da:.1%}'} "
            f"at {nm['test_coverage']:.0%} coverage (n={nm['test_n_dir']}) · "
            f"{'**beats** naive' if nm.get('beats_naive') else '**does not beat** naive'}. "
            "Prefer this number when the two disagree.")

# ---- chart ---------------------------------------------------------------------------
st.divider()
st.markdown("### Price history & forecast")
cc1, cc2, cc3 = st.columns([1, 1, 2])
show_all = cc1.checkbox("All horizons", value=True)
show_model = cc2.checkbox("Model track record", value=True,
                          help="At each date, what the model predicted for it h steps earlier.")
lookback = cc3.slider("History window (observations)", 60, 600, 240, step=20)

hist_d = live["history_dates"].iloc[-lookback:]
hist_p = live["history_prices"].iloc[-lookback:]
window_start = hist_d.iloc[0]

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_d, y=hist_p, name="Quoted price", mode="lines",
                         line=dict(color="#4a9eff", width=2),
                         hovertemplate="%{x|%b %d, %Y}<br>Quoted: %{y:,.2f}<extra></extra>"))

if show_model:
    track = predict_history(arts[headline_h], f"{choice}_h{headline_h}_{use_live}_{live_price}",
                            d_active)
    if len(track):
        tv = track[track.date >= window_start]
        if len(tv):
            fig.add_trace(go.Scatter(
                x=tv.date, y=tv.model, name=f"Model (in-arrears, h={headline_h})", mode="lines",
                line=dict(color="#e67e22", width=1.6, dash="dot"),
                hovertemplate="%{x|%b %d, %Y}<br>Model: %{y:,.2f}<extra></extra>"))
    try:
        tstart = pd.Timestamp(metas[headline_h]["meta"]["test_start"])
        if tstart >= window_start:
            fig.add_vline(x=tstart, line=dict(color="#ffffff", width=1.5, dash="dash"), opacity=0.65)
            fig.add_annotation(x=tstart, yref="paper", y=1.02, showarrow=False,
                               text="test set begins →", font=dict(size=11, color="#cbd5e1"),
                               xanchor="left")
    except Exception:
        pass

plot_hs = horizons if show_all else [headline_h]
fwd = [(pd.Timestamp(preds[h]["target_date"]), preds[h]["pred_price"], h)
       for h in plot_hs if preds[h]]
if fwd:
    fwd.sort()
    fig.add_trace(go.Scatter(
        x=[pd.Timestamp(live["last_date"])] + [f[0] for f in fwd],
        y=[live["last_price"]] + [f[1] for f in fwd],
        name="Forecast", mode="lines+markers",
        line=dict(color="#ff8c1a", width=2.4), marker=dict(size=10),
        text=["anchor"] + [f"h={f[2]}" for f in fwd],
        hovertemplate="%{text}<br>%{x|%b %d, %Y}<br>%{y:,.2f}<extra></extra>"))
if ff:
    fig.add_trace(go.Scatter(
        x=[pd.Timestamp(live["last_date"]) + pd.Timedelta(days=float(req_days))], y=[ff["value"]],
        name=f"{ff['mode'].title()} ({req_days}d)", mode="markers",
        marker=dict(size=13, symbol="diamond-open",
                    color="#dc2626" if ff["mode"] == "extrapolated" else "#a855f7",
                    line=dict(width=2)),
        hovertemplate=f"{ff['mode']}<br>%{{x|%b %d, %Y}}<br>%{{y:,.2f}}<extra></extra>"))

fig.update_layout(height=460, hovermode="x unified", margin=dict(t=44, b=10),
                  yaxis_title=unit, legend=dict(orientation="h", y=1.12, x=0),
                  xaxis=dict(showgrid=False))
st.plotly_chart(fig, use_container_width=True)
if show_model:
    st.caption("Dotted line = the model's own track record **in arrears**. Left of the dashed "
               "line the model trained on those rows (in-sample fit); only the segment to the "
               "**right** is genuine out-of-sample performance.")

# ---- COMPARE MODELS ------------------------------------------------------------------
st.divider()
st.markdown("### Compare model families")
st.caption("Overlay several families against the actual price at the headline horizon. The "
           "persistence baseline is always included — if a model's line sits on top of it, the "
           "model is not adding forecasting value.")

avail_h = sorted(models[choice].get(headline_h, {}))
picked = st.multiselect("Families to overlay", options=avail_h,
                        default=[family] + [f for f in avail_h if f != family][:2],
                        key=f"cmp_{choice}_{headline_h}")

if picked:
    figc = go.Figure()
    figc.add_trace(go.Scatter(x=hist_d, y=hist_p, name="Quoted price", mode="lines",
                              line=dict(color="#4a9eff", width=2.4)))
    palette = ["#e67e22", "#a855f7", "#16a085", "#dc2626", "#eab308", "#0ea5e9", "#f472b6"]
    cmp_rows = []
    for i, f in enumerate(picked):
        try:
            art_f = artifact_for(models[choice][headline_h][f])
            mt_f = meta_for(models[choice][headline_h][f])["metrics"]
            trk = predict_history(art_f, f"{choice}_{headline_h}_{f}_{use_live}_{live_price}",
                                  d_active)
            tvv = trk[trk.date >= window_start] if len(trk) else trk
            if len(tvv):
                figc.add_trace(go.Scatter(x=tvv.date, y=tvv.model, name=f, mode="lines",
                                          line=dict(color=palette[i % len(palette)], width=1.5,
                                                    dash="dot")))
            pf = predict_latest(art_f, d_active)
            cmp_rows.append({
                "family": f,
                "forecast": None if not pf else round(pf["pred_price"], 2),
                "% chg": None if not pf else round(pf["pct_change"], 2),
                "WF RMSE": round(mt_f.get("val_rmse", float("nan")), 2),
                "test RMSE": round(mt_f.get("test_rmse", float("nan")), 2),
                "test R²": round(mt_f.get("test_r2", float("nan")), 4),
                "DirAcc": None if mt_f.get("test_diracc") is None or pd.isna(mt_f["test_diracc"])
                          else round(mt_f["test_diracc"], 4),
                "beats persistence": mt_f.get("beats_naive", False),
                "DM p": round(mt_f.get("dm_p", float("nan")), 4),
            })
        except Exception as e:
            cmp_rows.append({"family": f, "forecast": None, "WF RMSE": None,
                             "test RMSE": None, "beats persistence": f"error: {e}"[:40]})

    pm = persistence_metrics(choice, headline_h, d_active)
    if pm:
        figc.add_trace(go.Scatter(
            x=[pd.Timestamp(live["last_date"]),
               pd.Timestamp(live["last_date"]) + pd.Timedelta(days=C.HORIZON_DAYS[headline_h])],
            y=[live["last_price"], live["last_price"]], name="Persistence (naive)",
            mode="lines+markers", line=dict(color="#94a3b8", width=2, dash="dash")))
        cmp_rows.append({"family": "PERSISTENCE (naive)", "forecast": round(live["last_price"], 2),
                         "% chg": 0.0, "WF RMSE": None,
                         "test RMSE": round(pm["test_rmse"], 2), "test R²": round(pm["test_r2"], 4),
                         "DirAcc": None, "beats persistence": "—", "DM p": None})

    figc.update_layout(height=430, hovermode="x unified", margin=dict(t=30, b=10),
                       yaxis_title=unit, legend=dict(orientation="h", y=1.12, x=0),
                       xaxis=dict(showgrid=False))
    st.plotly_chart(figc, use_container_width=True)
    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
    st.caption("A family whose **test RMSE** is not below the PERSISTENCE row is not forecasting — "
               "it is reproducing today's price. Judge on test RMSE and DM p, not on how the "
               "line looks.")

# ---- model health --------------------------------------------------------------------
st.divider()
st.markdown("### Model health — historical walk-forward performance")
st.caption("These are frozen numbers from the saved artifact. They describe the model's track "
           "record and do **not** change when you enter a live price.")

hm = metas[headline_h]
mt, mm = hm["metrics"], hm["meta"]
if mt.get("bh_significant"):
    st.success(f"**Stage-6 ACCEPTED.** Beats naive and survives Diebold-Mariano + "
               f"Benjamini-Hochberg multiple-comparison correction "
               f"(walk-forward DM p={mt.get('wf_dm_p', float('nan')):.4f}, "
               f"skill {mt.get('wf_skill', float('nan')):.3f}).")
elif mt.get("bh_significant") is False and mt.get("stage0_verdict") == "control":
    st.warning("Stage 0 found no statistically detectable structure in this series, and this "
               "configuration did not survive multiple-comparison correction. Treat any "
               "apparent edge here as unproven.")
if not mt.get("beats_naive", False):
    st.error(f"**This model does not beat the naive baseline at h={headline_h}.** "
             f"Test RMSE {mt['test_rmse']:,.2f} vs naive {mt['naive_rmse']:,.2f}. "
             "Treat its forecasts as unreliable.")
elif not mt.get("beats_naive_significant", False):
    st.warning(f"Beats naive (RMSE {mt['test_rmse']:,.2f} vs {mt['naive_rmse']:,.2f}) but "
               f"**not statistically significantly** (Diebold-Mariano p={mt['dm_p']:.3f}).")
else:
    st.success(f"Beats naive significantly — RMSE {mt['test_rmse']:,.2f} vs "
               f"{mt['naive_rmse']:,.2f} (Diebold-Mariano p={mt['dm_p']:.4f}).")

h1, h2, h3, h4 = st.columns(4)
h1.metric("Walk-forward RMSE", f"{mt['val_rmse']:,.2f}")
h2.metric("Test R²", f"{mt['test_r2']:.3f}")
h3.metric("Test dir. accuracy",
          "n/a" if mt.get("test_diracc") is None or pd.isna(mt["test_diracc"])
          else f"{mt['test_diracc']:.1%}")
h4.metric("Model family", mm["winner"])

with st.expander("Full metrics — every horizon"):
    rows = []
    for h in horizons:
        m, m2 = metas[h]["metrics"], metas[h]["meta"]
        rows.append({"h": h, "≈days": C.HORIZON_DAYS[h], "model": m2["winner"],
                     "target": m2["target_kind"], "RMSE": round(m["test_rmse"], 2),
                     "MAE": round(m["test_mae"], 2), "R²": round(m["test_r2"], 4),
                     "DirAcc": None if m["test_diracc"] is None or pd.isna(m["test_diracc"]) else round(m["test_diracc"], 4),
                     "Coverage": round(m["test_coverage"], 3), "N_dir": m["test_n_dir"],
                     "naive RMSE": round(m["naive_rmse"], 2), "beats naive": m["beats_naive"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("Confusion matrix (frozen operating point)"):
    cm = mt.get("test_cm")
    if cm:
        st.dataframe(pd.DataFrame(cm, index=["actual DOWN", "actual UP"],
                                  columns=["pred DOWN", "pred UP"]), use_container_width=True)
        tn, fp = cm[0]
        fn, tp = cm[1]
        tot = tn + fp + fn + tp
        if tot:
            st.caption(f"n={tot}. Always guessing the majority class would score "
                       f"{max(tn + fp, fn + tp) / tot:.1%} — compare the model's "
                       f"{mt['test_diracc']:.1%} against that, not against 50%.")
    else:
        st.write("Not enough directional observations at this operating point.")

with st.expander("Training configuration & provenance"):
    st.json({k: mm[k] for k in ("resource", "horizon", "approx_days", "target_kind", "winner",
                                "strategy", "dead_zone", "n_samples", "n_features",
                                "train_start", "train_end", "test_start", "test_end",
                                "n_train", "n_val", "n_test")})
    st.markdown("**Top features (train-only consensus ranking)**")
    st.write(", ".join(mm.get("top_features", [])[:15]))
