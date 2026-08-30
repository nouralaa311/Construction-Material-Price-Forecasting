# Egypt-Enhanced Steel Price Dataset — Merge & Audit Summary

**Artifacts produced**
- `steel_data_egypt_combined.csv` — the combined dataframe (983 rows × 71 columns)
- `steel_data_egypt_combined_nullreport.csv` — per-column null breakdown (leading vs interior)
- `Steel_Price_Forecasting_Egypt_Enhanced.ipynb` — the notebook that builds, audits and models it

---

## 1. Sources joined

Every join onto the base date index uses `pandas.merge_asof(direction='backward')` against an
explicit `available_date` — never the nominal period date, and never a plain date merge.

| # | Source | Series / Indicator | Frequency | Publication lag applied | Coverage |
|---|---|---|---|---|---|
| 1 | Base steel series | `allprices.csv` (`Steel_Price`, `USD_Sell`, Oil/Aluminium/Copper/Cement) | daily-ish, irregular (1–39 day gaps) | n/a (the base index) | 2023-01-11 → 2026-08-03 |
| 2 | FRED / IMF | `PIORECRUSDM` — global iron ore price | monthly | **+35 days** | 1990-01 → 2026-07 |
| 3 | FRED / IMF | `PCOALAUUSDM` — Australian thermal coal | monthly | **+35 days** | 1990-01 → 2026-07 |
| 4 | FRED / IMF | `PNGASUSUSDM` — Henry Hub natural gas | monthly | **+35 days** | 1990-01 → 2026-07 |
| 5 | World Bank | `FP.CPI.TOTL.ZG` — Egypt CPI inflation | annual | **treated as available 1 Jul of year+1 (~6 months)** | 1961 → 2025 |
| 6 | World Bank | `NY.GDP.MKTP.KD.ZG` — Egypt GDP growth | annual | **treated as available 1 Jul of year+1 (~6 months)** | 1961 → 2025 |
| 7 | countryeconomy.com | CBE key policy rate (official MPC decisions) | irregular step series (~6–8/yr) | **0 days** (announced publicly same day) | 2022-10-27 → 2026-02-15 |

### Source URLs / citations
- Iron ore: `https://fred.stlouisfed.org/series/PIORECRUSDM` (IMF Primary Commodity Price System)
- Coal: `https://fred.stlouisfed.org/series/PCOALAUUSDM` (IMF)
- Natural gas: `https://fred.stlouisfed.org/series/PNGASUSUSDM` (IMF)
- Egypt CPI: `https://api.worldbank.org/v2/country/EG/indicator/FP.CPI.TOTL.ZG`
- Egypt GDP: `https://api.worldbank.org/v2/country/EG/indicator/NY.GDP.MKTP.KD.ZG`
- CBE policy rate: `https://countryeconomy.com/key-rates/egypt` — used because `cbe.org.eg`
  returns a WAF block to automated requests; every figure corresponds to a publicly announced
  CBE Monetary Policy Committee decision.

### Why the publication lag matters
A "March" FRED monthly value is not knowable on 1 March. Joining on the nominal period date would
hand the model information weeks before it existed. Each external table therefore carries
`available_date = period_date + lag`, and `merge_asof` matches on that. CBE decisions get a 0-day
lag because they are announced publicly the day they take effect.

---

## 2. Explicitly EXCLUDED — investigated, not obtainable, **not fabricated**

| Requested source | Why it is absent |
|---|---|
| CAPMAS Producer Price Index | No API or machine-readable series. Only individual monthly figures embedded in news coverage of press releases. |
| CAPMAS construction-materials price index | PDF bulletins only, no downloadable series. |
| Egypt industrial production / manufacturing index | No accessible source found, official or aggregator. |
| EGX stock index | Aggregators block automated access (`investing.com` → HTTP 403) or prohibit scraping in their terms. |
| Egypt steel/iron trade volumes (HS72) | UN Comtrade's free tier is genuine and returned real annual totals for 2022 and 2024, but 2023 was empty — two usable points across ~920 rows is not a usable feature. |
| CBE daily official EGP rate | `cbe.org.eg` WAF-blocks automated fetches. The base dataset's own `USD_Buy`/`USD_Sell` columns already provide a higher-frequency EGP-rate proxy. |

No placeholder or synthetic values were generated for any of the above.

---

## 3. Derived feature groups (51 Egypt columns after pruning)

| Group | Count | Content |
|---|---|---|
| EGP-adjusted commodity costs | 18 | `oil_egp`, `alu_egp`, `cop_egp` (= USD price × `USD_Sell`), their 5/21-period returns, 21-period volatility, steel-to-input ratios and 21-period z-scores |
| Global input costs | 24 | iron ore / coal / natgas level, `ret_1m`, `ret_3m`, `vol_6m`, `mom_12m`, `mean_3m`, `dev_12m`, EGP-adjusted level, steel-to-input ratio |
| CBE policy rate | 3 | `cbe_rate`, `cbe_rate_change_90`, `cbe_days_since_change` |
| Egypt macro backdrop | 4 | CPI inflation + change, GDP growth + change |

Commodity return/volatility statistics are computed **on the true monthly cadence before the
as-of merge** — computing them on the forward-filled daily step function would badly distort
volatility (long runs of zero change).

---

## 4. Verification results

### 4a. Row-count integrity — **PASS**
```
base rows      : 983
combined rows  : 983
```
A backward as-of join must neither create nor drop base rows. Date ordering and uniqueness also
asserted.

### 4b. No-leakage assertions — **PASS (both)**

1. **`assert_egypt_merge_safe()`** — poisons every external table (iron ore, coal, gas, CBE rate)
   by multiplying all values after a 2025-01-01 cutoff by a random 3–9×, rebuilds the features,
   and asserts every row up to the cutoff is bit-identical. This is the test that specifically
   targets `merge_asof` / `available_date` correctness.
2. **`assert_causal(build_features_full, df)`** — scrambles all base-data rows after row 600 and
   asserts all **371** engineered features (original + Egypt) are unchanged in the history.

A static source scan additionally rejects `bfill`, `backfill`, `center=True` and negative shifts
anywhere in the feature-engineering code.

### 4c. Null-check summary — **PASS, no unexpected interior holes**

Every interior null was traced to its input columns and verified to match **exactly** (actual ==
expected) across all 19 derived families, with the correct lag semantics
(`ret_k` is NaN iff row *t* or row *t−k* is NaN, and the first *k* rows are NaN by construction).

The 7 external step/monthly series have **zero interior nulls**:
`cbe_rate`, `cbe_days_since_change`, `iron_ore_level`, `coal_level`, `natgas_level`,
`egypt_cpi_inflation`, `egypt_gdp_growth`.

### 4d. Redundancy / leakage-smell screen — **clean**
- exact duplicate column pairs: **0**
- near-zero-variance columns: **0**
- `|corr|` with target > 0.95 (level space): **0**
- `|corr|` with target > 0.95 (return space — the honest test): **0**

---

## 5. Improvement pass applied

### Causal forward-fill of `USD_Sell` (the big one)
`USD_Sell` was missing on **33.5%** of rows, and every EGP-adjusted feature multiplies through it,
propagating that hole into **46–71% missingness across 20 features**.

Forward-fill is the economically correct treatment for a step-persistent quote (the last posted
rate *is* the prevailing rate until a new one is posted) and is **causal by construction** — it
only ever carries a past value forward. Result:

| Feature | Before | After |
|---|---|---|
| `oil_egp` | 46.3% missing | **0.0%** |
| `alu_egp` | 47.6% missing | **0.0%** |
| `cop_egp` | 46.6% missing | **0.0%** |
| `iron_ore_egp` / `coal_egp` / `natgas_egp` | 33.5% missing | **0.0%** |

Overall the Egypt block went from 25 columns with interior nulls to 10, and the worst case fell
from 71% to 1.2%. The residual nulls are the 5 rows where `Steel_Price` itself is missing — rows
that are excluded from supervised samples regardless.

The scramble assertion was re-run after this change and still passes, confirming the ffill
introduced no look-ahead.

### Features added
`mean_3m` and `dev_12m` (deviation from 12-month mean) for each of iron ore, coal and natural gas
— computed on the true monthly cadence.

### Features dropped — train-only redundancy prune
3 features removed as pairwise-redundant (train |r| > 0.97), keeping whichever sibling had the
stronger train-only association with the target return:
`alu_egp_vol_21`, `egp_devaluation_flag`, `steel_cop_egp_ratio`.

> **Important:** the redundancy screen is computed on **training rows only**. Measuring
> correlations over the full sample (test included) and using them to decide which columns survive
> would be a quiet form of test-set peeking. Weak-signal features are also deliberately *not*
> hand-dropped — Section 11's consensus ranking (MI + |corr| + XGB gain + permutation + SHAP, all
> train-only) is the mechanism that decides what earns a place in the model.

### Strongest Egypt signals (full-sample |r| vs target log-return, diagnostic only)
| Feature | \|r\| |
|---|---|
| `steel_alu_egp_ratio_z21` | 0.238 |
| `steel_cop_egp_ratio_z21` | 0.226 |
| `steel_oil_egp_ratio_z21` | 0.226 |
| `alu_egp_ret_5` | 0.102 |
| `iron_ore_ret_3m` | 0.082 |
| `steel_iron_ore_egp_ratio` | 0.078 |

Median |r| across all Egypt features is 0.028 — against a near-random-walk return series, the
z-scored steel-to-input-cost ratios standing at 0.22–0.24 is the most promising signal in the set.
