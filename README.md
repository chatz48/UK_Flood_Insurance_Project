# UK Flood Insurance Catastrophe Model

A probabilistic CAT model for UK residential flood risk built from publicly available data. The goal is to produce a credible Loss Exceedance Curve (LEC) and Annual Average Loss (AAL) for a residential portfolio, validated against observed insured losses from UK flood events since 2000.

---

## What this project does

I've built this as a full four-layer catastrophe model following the structure used in commercial cat modelling:

```
Hazard → Exposure → Vulnerability → Loss Aggregation
```

- **Hazard** — GEV distributions fitted to NRFA annual maximum flows for ~880 gauging stations, giving return period flows at T=2 through T=1000 years
- **Exposure** — Land Registry price paid data geocoded to lat/lon, with EA flood zone assignment and property value aggregation by postcode
- **Vulnerability** — DEFRA FD2320 residential depth-damage curve with adjustments for contamination and flood duration
- **Loss** — 10,000 stochastic Monte Carlo events producing a full LEC, AAL, and TVaR (99.5%)

There's also a Thames RDS scenario benchmarked against the Lloyd's £6.2bn industry loss figure, climate-adjusted AALs under UKCP18 RCP8.5, and a TIV accumulation table by flood zone and county.

---

## Data

All data is publicly available and free. The pipeline scripts download and cache everything automatically.

| Source | Used for |
|---|---|
| NRFA (CEH) | Peak flow records for hazard layer |
| HM Land Registry Price Paid | Property values and postcode exposure (2020–2024) |
| EA Flood Zones (Flood Map for Planning) | Flood zone 2/3 assignment per property |
| VOA Council Tax Stock | Property count by local authority |
| IMD 2019 (MHCLG) | Deprivation index as vulnerability modifier |
| EA Real-Time Monitoring API | Gauging station metadata |
| ABI / DEFRA flood events | 13 observed events for model validation |
| NFIP Claims (FEMA) | US claims data used as damage function calibration reference |

**Note on EA flood zones:** the EA ArcGIS REST API for the statutory flood zone boundaries is currently returning errors. This needs a manual download from [data.gov.uk](https://www.data.gov.uk/dataset/2a6f4a16-31c7-4cf2-a843-ec80bc7e88af) and placing the file at `data/raw/ea_flood_zones/rofrs_zones_by_region.parquet`. Without it, properties default to flood zone "none" and the spatial loss allocation is effectively disabled.

---

## Running it

```bash
pip install -r requirements.txt

# Download all data (cached, re-runnable)
python run_pipelines.py

# Build geocoded exposure portfolio (~20 min, parallel)
python src/exposure/portfolio.py

# Run model — prints val_score on last line
python train.py
```

---

## Outputs

- `outputs/results/loss_exceedance_curve.parquet` — full LEC
- `outputs/results/climate_aal.json` — baseline + 2030/2050/2080 AAL under RCP8.5
- `outputs/results/thames_rds_scenario.json` — Thames corridor RDS vs Lloyd's benchmark
- `outputs/results/tiv_by_zone.parquet` — TIV split by flood zone and property type

---

## Known limitations and what's needed to fix them

**The biggest gaps right now:**

1. **EA flood zone polygons** — without these, there's no spatial loss allocation. Everything is modelled at portfolio level rather than property level. Getting the RoFRS shapefile is the single highest-impact improvement.

2. **TIV is inflated** — I'm using Land Registry `transaction_count` as a proxy for property count, which overcounts. The realistic residential TIV for England should be around £400–600bn; the model currently produces ~£2,160bn. The fix is to use VOA council tax stock counts as the property count denominator.

3. **No DEM** — flood depth is estimated from a simplified Manning's-law approximation rather than actual elevation data. OS Terrain 50 is free under PSGA and would substantially improve depth estimation.

4. **Spatial correlation** — a single scalar factor is used across the whole portfolio. Proper event footprints (flood extent polygons per event) are needed for realistic tail loss behaviour.

**On methodology:**

The hazard layer uses GEV MLE fitted station-by-station. The industry standard for UK hydrology is FEH regional pooling (pooled growth curves across hydrologically similar stations), which gives more stable parameter estimates for stations with short records. I've stubbed this in as `AMAX_REGIONAL_POOLING` in `train.py` but haven't implemented it yet.

The DEFRA FD2320 damage function is the standard for UK residential flood damage assessment, so that part is fit for purpose. The depth offset and contamination parameters are calibrated to minimise val_score against the 13 known events — but there aren't enough events to do a proper train/test split, so the validation is effectively in-sample.

A proper commercial implementation would link each property to a river reach, run a hydraulic model to get flood extents per return period, and apply the damage function property by property. What I've built is a portfolio-level approximation that captures the right order of magnitude but can't produce spatially granular results.
