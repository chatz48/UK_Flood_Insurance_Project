# UK Flood Insurance Cat Model — AutoResearch Program

## Objective
Minimise `val_score` — a composite metric measuring how accurately the model
reproduces the known insured loss magnitudes of 13 major UK flood events
(2000–2024) compared to their empirical return periods.

**val_score** (lower is better) = log_RMSE + 0.5 * |bias| + 0.3 * tail_log_RMSE

Tail accuracy is weighted because the 1-in-100 and 1-in-200 year losses
are what regulators and reinsurers price against.

---

## Files You Can Modify

You have TWO modes of operation. Both use the same keep/discard loop.

### Mode 1 — Model tuning (fast, ~1-2 min per experiment)
Modify `train.py` only. Change parameters, distributions, damage curves,
simulation logic. This is the default mode and should be your first priority.

### Mode 2 — New data source (slow, ~5-15 min per experiment)
When you believe a new data source could improve val_score:
1. **Search** for the dataset using WebSearch or WebFetch
2. **Write** a new pipeline in `src/pipelines/<source_name>.py`
   — copy `src/pipelines/pipeline_template.py` as your starting point
3. **Run** it: `python src/pipelines/<source_name>.py`
   — output goes to `data/features/<feature_name>.parquet`
4. **Register** it in `src/features/feature_registry.py` → FEATURE_CATALOGUE
5. **Activate** it by adding the name to ACTIVE_FEATURES in `train.py`
6. **Run** `python train.py` — does val_score improve?
7. Keep or discard the whole set of changes together

---

## Setup (once per run)
1. Agree on run tag (e.g. `mar22`). Create branch: `git checkout -b autoresearch/<tag>`
2. Read `train.py`, `src/features/feature_registry.py`, and this file fully
3. Check `data/raw/nrfa/` and `data/raw/abi_events/` are populated
4. Run `python autoresearch/results_init.py` to create `autoresearch/results.tsv`
5. Check what features are already available: `python src/features/feature_registry.py`
6. Confirm, then begin

---

## Experiment Loop (NEVER STOP — run until manually interrupted)

1. Read current `train.py` + `autoresearch/results.tsv` for context
2. Choose ONE experiment (Mode 1 or Mode 2)
3. Make changes, then: `git add -A && git commit -m "experiment: <description>"`
4. `python train.py > run.log 2>&1`
5. `grep "^val_score:" run.log`
6. If empty → crashed. `tail -50 run.log`. Fix if trivial, else discard + revert
7. Log to `autoresearch/results.tsv`:
   `<7-char-commit>\t<val_score>\t<keep|discard|crash>\t<description>`
8. If val_score improved → keep commit, advance
9. If equal or worse → `git reset --hard HEAD~1`
10. **NEVER ask the human if you should continue**

---

## Model Parameters to Tune (train.py)

### Hazard Layer
- `HAZARD_FITTING_METHOD`: "gev_lmom" | "gev_mle" | "gumbel_lmom"
  GEV with L-moments is the FEH standard. MLE may outperform for records >30 years.
  Gumbel assumes thin tails — UK floods have heavy tails, so GEV likely wins.
- `MIN_RECORD_YEARS`: 10–30
  Fewer years = more stations but worse-fitted distributions.
  More years = fewer stations but better tail estimates. Explore the trade-off.
- `AMAX_REGIONAL_POOLING`: False | True
  FEH recommends regional pooling for short records. May help or introduce bias.

### Vulnerability Layer
- `DAMAGE_FUNCTION`: "defra_fd2320" | "exponential" | "sigmoid" | "piecewise"
  DEFRA FD2320 is calibrated on 1990s housing stock. Modern homes may differ.
  Sigmoid has an inflection point that may better fit observed damage surveys.
- `DAMAGE_DEPTH_OFFSET`: -0.3 to +0.3 metres
  Positive = more damage at given depth (older stock, no resilience measures)
- `CONTAMINATION_RATE`: 0.0–0.6 (urban combined sewer systems ~0.4)
- `MEAN_FLOOD_DURATION_DAYS`: 0.5–10.0
  Cumbria 2015 = 7+ days, Boscastle 2004 = hours. National average ~2-3 days.
- Exponential curve: `EXP_K` (0.4–2.0), `EXP_SATURATION` (0.5–0.95)
- Sigmoid curve: `SIG_MIDPOINT` (0.3–1.5m), `SIG_STEEPNESS` (1.0–6.0), `SIG_MAX_DAMAGE` (0.6–0.95)

### Exposure Layer
- `PROPERTY_VALUE_SOURCE`: "median" | "mean"
- `PROPERTY_VALUE_INFLATION`: 1.0–1.4 (market price to rebuild cost ratio)
- `DEFAULT_PROPERTY_VALUE_GBP`: 150000–350000

### Simulation
- `N_STOCHASTIC_EVENTS`: 5000–100000
- `SPATIAL_CORRELATION_FACTOR`: 0.5–1.0
  Spatial correlation is typically underestimated. Try 0.90–0.95.

---

## New Data Sources to Explore (Mode 2)

These are CANDIDATE features in `src/features/feature_registry.py`.
For each, the fetch_notes field tells you where to get it.
Prioritise by expected signal strength:

### HIGH PRIORITY — likely high signal
1. **council_tax_bands** — VOA council tax distribution per postcode
   - Direct CSV download, no auth: `voa.gov.uk/corporate/datasets/council-tax-stock-of-properties.html`
   - Gives property value distribution better than Land Registry (includes non-transacted properties)
   - Expected benefit: improves exposure layer accuracy

2. **postcode_deprivation** — Index of Multiple Deprivation (IMD)
   - Direct download: `gov.uk/government/statistics/english-indices-of-deprivation-2019`
   - Correlates with property age, flood resilience, repair costs
   - Expected benefit: adjusts vulnerability curve by area type

3. **ea_gauge_statistics** — compute exceedance days from EA API historical readings
   - Data already accessible via the EA gauging pipeline we built
   - Compute: days above flood alert / warning threshold per year per station
   - Expected benefit: better calibrates the hazard-to-loss scaling

4. **ceh_soil_wetness** — UKCEH Soil Wetness Index
   - Access via UKCEH Environmental Information Platform (free registration)
   - Antecedent soil moisture strongly predicts flood peak amplification
   - Expected benefit: improves storm event loss estimates

### MEDIUM PRIORITY — moderate signal, more effort
5. **ukcp18_climate** — Met Office UKCP18 climate projections
   - Download from CEDA archive (free, requires registration at ceda.ac.uk)
   - Gives future flood frequency uplift factors (e.g. +20% flows by 2050 under RCP8.5)
   - Expected benefit: allows forward-looking validation; may improve model calibration

6. **era5_precipitation** — Copernicus ERA5 reanalysis rainfall
   - `pip install cdsapi`, register at cds.climate.copernicus.eu
   - 80 years of hourly UK rainfall → independent flood frequency estimates
   - Expected benefit: supplements NRFA gauge data for ungauged catchments

7. **historic_flood_claims_hull** — 2007 Hull post-event damage survey
   - Extract tables from the Hull City Council 2007 flood report (PDF, publicly available)
   - Property-level damage fractions — direct vulnerability curve calibration data
   - Expected benefit: could significantly improve damage function fitting

### LOW PRIORITY — search for more
8. Search for any of the following that you find through WebSearch:
   - "UK flood claims data CSV"
   - "NFIP equivalent UK flood insurance data"
   - "Environment Agency flood damage assessment survey"
   - "Flood Re actuarial data"
   - Post-event surveys for: Carlisle 2005, Somerset 2013/14, Cumbria 2015, Yorkshire 2019

---

## Constraints (NEVER VIOLATE)

1. AAL must be £300m–£1.5bn (ABI benchmark ~£500-800m average annual)
2. 1-in-200 year loss must be £3bn–£12bn (Solvency II plausibility range)
3. Loss Exceedance Curve must be monotonically decreasing (physically required)
4. val_score must be finite (not NaN or inf)
5. Model must complete within 5 minutes
6. Do not modify `src/metrics/return_periods.py` — the scoring function is fixed
7. All new pipeline scripts must save output to `data/features/` as parquet
8. New pipeline scripts must not break if their data source is unavailable

---

## Simplicity Criterion

All else equal, simpler is better.
- A 0.01 val_score improvement that adds 50 lines of hacky code → probably not worth it
- Removing an adjustment and getting equal/better val_score → always keep (simplification win)
- A new data source that adds 30 lines but improves val_score by 0.05 → worth it

---

## Known Calibration Points

The 2007 UK floods (£3.2bn insured, ~1-in-35 year event) is the single most
important calibration point. A good model places the 1-in-35 year loss
between £2.5bn and £4.5bn.

Return period source: Marsh & Hannaford (2007) CEH/NERC hydrological appraisal
gives 30–45 years for peak flows on the Severn. The previously used T=75 figure
was not supported by published hydrological analysis.

ABI annual aggregate benchmark: ~£500–800m average annual flood loss.
Your AAL must fall in this range.

If you get stuck, re-read the full list of kept experiments in results.tsv,
then try combining the best two near-misses, or attempt a more radical change
to the simulate_event_losses() function in train.py.
