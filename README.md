# UK Flood Insurance Catastrophe Model

A probabilistic catastrophe (CAT) model for UK residential flood risk. Estimates annual expected losses, return period losses, and scenario losses for an insured property portfolio.

---

## Objective

Produce a **Loss Exceedance Curve (LEC)** and **Annual Average Loss (AAL)** for a UK residential flood portfolio, validated against observed insured losses from known flood events (2000–2024). The model follows the standard four-layer CAT modelling chain used in the insurance and reinsurance industry.

---

## Architecture

```
Hazard  →  Exposure  →  Vulnerability  →  Loss Aggregation
```

| Layer | What it does | Key file |
|---|---|---|
| **Hazard** | Fits GEV distributions to NRFA peak flows; estimates return period flows per gauging station | `src/hazard/flood_frequency.py` |
| **Exposure** | Builds property portfolio from Land Registry; geocodes postcodes; assigns EA flood zones | `src/exposure/portfolio.py` |
| **Vulnerability** | Applies DEFRA FD2320 depth-damage curve to convert flood depth to damage fraction | `src/vulnerability/damage_functions.py` |
| **Loss** | Monte Carlo simulation over 10,000 events; builds LEC and computes AAL | `src/metrics/return_periods.py` |

Supporting modules: `src/scenarios/thames_rds.py` (Lloyd's RDS scenario), `src/analysis/tiv_accumulation.py` (exposure aggregation), `src/features/feature_registry.py` (plug-in data sources).

---

## Data Sources

| Source | What it provides | Status |
|---|---|---|
| NRFA (CEH) | Annual maximum flows for ~1,500 gauging stations | Downloaded |
| HM Land Registry Price Paid | Property values and transaction counts per postcode (2020–2024) | Downloaded |
| Environment Agency Flood Zones | Statutory flood zone 2/3 boundaries | Partial — API unavailable, manual download required |
| VOA Council Tax Stock | Property count and band distribution per local authority | Downloaded |
| IMD 2019 (MHCLG) | Deprivation scores per LSOA | Downloaded |
| EA Real-Time Monitoring API | Gauging station metadata and thresholds | Downloaded |
| ABI / DEFRA flood events | Observed insured losses for model validation (13 events) | Included |
| NFIP Claims (FEMA) | US flood claims — used for damage function calibration reference | Downloaded |

Run `python run_pipelines.py` to download all sources. Re-runnable; uses local caches.

---

## Outputs

| Output | Location |
|---|---|
| Loss Exceedance Curve | `outputs/results/loss_exceedance_curve.parquet` |
| Return period losses (T=10 to T=1000) | Printed by `train.py` |
| Annual Average Loss (AAL) | Printed by `train.py` |
| Climate-adjusted AAL (UKCP18 RCP8.5) | `outputs/results/climate_aal.json` |
| Thames RDS scenario (vs Lloyd's £6.2bn benchmark) | `outputs/results/thames_rds_scenario.json` |
| TIV accumulation by flood zone / county | `outputs/results/tiv_by_zone.parquet` |

Run: `python train.py` — prints `val_score` on the last line (lower is better).

---

## Current Limitations

**Data gaps:**

- **EA Flood Zone polygons** — The statutory Flood Map for Planning (Zones 2/3a/3b) is unavailable via API; the EA ArcGIS REST endpoints are down. Without this, every property defaults to flood zone "none" and the spatial loss allocation is missing. Manual download from [data.gov.uk](https://www.data.gov.uk/dataset/2a6f4a16-31c7-4cf2-a843-ec80bc7e88af) required.
- **Property counts** — Land Registry `transaction_count` counts *transactions*, not *properties*. This inflates portfolio TIV (currently ~£2,160bn vs realistic ~£400–600bn for residential England). Replacing with VOA council tax stock counts per postcode would fix this.
- **Flood depth** — No DEM (Digital Elevation Model) data is integrated. Flood depth is estimated from a simplified Manning's-law stage-discharge approximation. OS Terrain 50 (free under PSGA) would improve this substantially.
- **Spatial correlation** — A single scalar correlation factor is applied across the portfolio. A proper event footprint (spatial flood extent polygon per event) is needed for realistic tail loss estimation.

**Methodology:**

- **Hazard**: GEV MLE fitting per station. Industry standard is regional pooling (FEH methods — pooled growth curves) which reduces parameter uncertainty for short-record stations. The `AMAX_REGIONAL_POOLING` flag exists but is not yet implemented.
- **Exposure → Hazard link**: Properties are not individually linked to gauging stations or flood zones. Loss is computed as a portfolio-level fraction, not property-by-property. A full implementation would assign each property to a river reach and compute depth from a hydraulic model.
- **Damage function**: DEFRA FD2320 is the UK industry standard for residential flood damage, so this is appropriate. The calibrated parameters (`depth_offset=0.15m`, `contamination_rate=0.50`) are reasonable but not independently validated against a held-out dataset.
- **Validation**: Only 13 labelled events are available (ABI/DEFRA). This is too few for robust out-of-sample validation of the tail. The model is calibrated and validated on the same event set, which risks overfitting.

---

## What Would Improve the Model

1. **EA Flood Zone 2/3 polygons** — single biggest gap; unlocks property-level flood zone assignment
2. **OS Terrain 50 DEM** — enables depth estimation from hydraulic flood levels rather than zone-based proxies
3. **FEH regional pooling** — improves return period flow estimates for short-record stations
4. **Property-level event footprints** — replace the portfolio-fraction approach with spatially explicit loss calculation
5. **Separate contents / buildings split** — ABI reporting combines both; splitting improves damage function calibration

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Download data (runs all pipelines with caching)
python run_pipelines.py

# Build exposure portfolio (geocodes ~1M postcodes, ~20 min with parallel workers)
python src/exposure/portfolio.py

# Run model and print val_score
python train.py
```
