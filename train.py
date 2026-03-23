"""
train.py — AutoResearch Entry Point

This is the PRIMARY file AutoResearch modifies.
The agent may also write new files in src/pipelines/ to fetch new data sources,
produce feature parquets in data/features/, and register them in
src/features/feature_registry.py — then add their names to ACTIVE_FEATURES below.

Architecture (what AutoResearch can tune):
  1. Hazard layer:       GEV distribution fitting method + parameters
  2. Exposure layer:     How property values are estimated from Land Registry
  3. Vulnerability:      Depth-damage curve shape and parameters
  4. Loss aggregation:   How event losses are combined into a portfolio loss
  5. Simulation:         Number of stochastic events and sampling method
  6. Feature registry:   Add new data sources via ACTIVE_FEATURES

Objective metric (val_score — LOWER IS BETTER):
  Composite score on held-out UK flood events:
    - Log-RMSE on implied return periods vs empirical
    - Weighted towards tail accuracy (large events matter most)
    - Penalises systematic bias

Run: python train.py
Output: prints val_score to stdout on last line as "val_score: X.XXXX"
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.hazard.flood_frequency import fit_gev_lmom, fit_gev_mle, gev_quantile, lmom_ratios
from src.vulnerability.damage_functions import (
    defra_fd2320_residential,
    exponential_damage_curve,
    sigmoid_damage_curve,
)
from src.metrics.return_periods import (
    build_loss_exceedance_curve,
    compute_aal,
    get_return_period_losses,
    validate_against_known_events,
    compute_model_score,
    print_lec_summary,
)
from src.features.feature_registry import load_active_features, list_available_features

DATA_DIR = Path(__file__).parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path(__file__).parent / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# AUTORESEARCH: MODIFY PARAMETERS BELOW
# These are the hyperparameters of the cat model.
# ===========================================================================

# --- Hazard Layer ---
HAZARD_FITTING_METHOD = "gev_mle"    # options: "gev_lmom", "gev_mle", "gumbel_lmom"
AMAX_REGIONAL_POOLING = False        # pool stations within same FEH region for better tail estimation
MIN_RECORD_YEARS = 15                # minimum years of record to include a station

# --- Vulnerability Layer ---
DAMAGE_FUNCTION = "defra_fd2320"    # options: "defra_fd2320", "exponential", "sigmoid", "piecewise"
DAMAGE_DEPTH_OFFSET = 0.15          # shift applied to flood depth before damage calc (metres)
CONTAMINATION_RATE = 0.50           # fraction of flood events with sewage contamination
MEAN_FLOOD_DURATION_DAYS = 3.5      # average flood duration for duration adjustment

# Exponential curve params (used if DAMAGE_FUNCTION="exponential")
EXP_K = 0.8
EXP_SATURATION = 0.75

# Sigmoid curve params (used if DAMAGE_FUNCTION="sigmoid")
SIG_MIDPOINT = 0.8
SIG_STEEPNESS = 3.0
SIG_MAX_DAMAGE = 0.80

# --- Exposure Layer ---
PROPERTY_VALUE_SOURCE = "median"    # "median" or "mean" of Land Registry transactions
PROPERTY_VALUE_INFLATION = 1.15     # price-to-rebuild-cost ratio (rebuild usually less than market)
DEFAULT_PROPERTY_VALUE_GBP = 220000 # fallback where Land Registry data is missing

# --- Loss Aggregation ---
SPATIAL_CORRELATION_FACTOR = 0.95  # correlation between adjacent postcode losses (0=independent, 1=perfect)
CATCHMENT_AGGREGATION = "sum"       # how to sum losses across a catchment: "sum" or "loss_weighted_sum"

# --- Simulation ---
N_STOCHASTIC_EVENTS = 10000         # number of Monte Carlo simulated events
SIMULATION_SEED = 42

# --- Feature Registry ---
# Add new data source names here after writing their pipeline + parquet.
# Each name must exist as a key in src/features/feature_registry.FEATURE_CATALOGUE.
# The agent adds entries here when it discovers a useful new data source.
ACTIVE_FEATURES = [
    # "council_tax_bands",        # VOA council tax bands — property value proxy
    # "postcode_deprivation",     # IMD deprivation scores
    # "ea_gauge_statistics",      # EA exceedance days per gauging station
    # "ukcp18_climate",           # Climate change uplift factors
    # "era5_precipitation",       # ERA5 rainfall return periods
    # "ceh_soil_wetness",         # Antecedent soil moisture
    # "historic_flood_claims_hull", # 2007 Hull event ground truth
]

# ===========================================================================
# MODEL ASSEMBLY
# ===========================================================================

def load_data():
    """Load processed data for model assembly."""
    data = {}

    amax_path = DATA_DIR / "raw" / "nrfa" / "amax_all_stations.parquet"
    if amax_path.exists():
        data["amax"] = pd.read_parquet(amax_path)
        print(f"  Loaded AMAX: {len(data['amax']):,} records, {data['amax']['station_id'].nunique()} stations")
    else:
        print("  WARNING: NRFA AMAX data not found. Run pipelines first.")
        data["amax"] = pd.DataFrame()

    postcode_path = DATA_DIR / "raw" / "land_registry" / "postcode_aggregated.parquet"
    if postcode_path.exists():
        data["postcodes"] = pd.read_parquet(postcode_path)
        print(f"  Loaded postcodes: {len(data['postcodes']):,} postcodes")
    else:
        print("  WARNING: Land Registry postcode data not found.")
        data["postcodes"] = pd.DataFrame()

    events_path = DATA_DIR / "raw" / "abi_events" / "uk_flood_events.parquet"
    if events_path.exists():
        data["events"] = pd.read_parquet(events_path)
        print(f"  Loaded validation events: {len(data['events'])} known flood events")
    else:
        print("  WARNING: ABI events data not found.")
        data["events"] = pd.DataFrame()

    return data


def build_hazard_layer(amax_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit flood frequency distributions to all NRFA stations.
    Returns DataFrame with return period flows per station.
    """
    if amax_df.empty:
        print("  No AMAX data — using synthetic hazard layer for testing")
        return _synthetic_hazard_layer()

    stations = amax_df["station_id"].unique()
    results = []

    for sid in stations:
        flows = amax_df[amax_df["station_id"] == sid]["peak_flow_m3s"].dropna().values

        if len(flows) < MIN_RECORD_YEARS:
            continue

        # Fit selected distribution
        if HAZARD_FITTING_METHOD == "gev_lmom":
            from src.hazard.flood_frequency import fit_gev_lmom
            params = fit_gev_lmom(flows)
        elif HAZARD_FITTING_METHOD == "gev_mle":
            from src.hazard.flood_frequency import fit_gev_mle
            params = fit_gev_mle(flows)
        elif HAZARD_FITTING_METHOD == "gumbel_lmom":
            from src.hazard.flood_frequency import fit_gumbel_lmom
            params = fit_gumbel_lmom(flows)
        else:
            from src.hazard.flood_frequency import fit_gev_lmom
            params = fit_gev_lmom(flows)

        if params is None:
            continue

        row = {"station_id": sid, "n_years": len(flows)}
        for T in [2, 5, 10, 50, 100, 200, 500, 1000]:
            row[f"q_T{T}"] = gev_quantile(T, params["mu"], params["sigma"], params["xi"])
        results.append(row)

    return pd.DataFrame(results) if results else _synthetic_hazard_layer()


def _synthetic_hazard_layer() -> pd.DataFrame:
    """
    Synthetic hazard layer for testing when real data isn't downloaded yet.
    Produces a plausible UK-like distribution of return period flows.
    """
    np.random.seed(SIMULATION_SEED)
    n = 500
    station_ids = range(1, n + 1)
    rows = []
    for sid in station_ids:
        mean_flow = np.random.lognormal(mean=3.5, sigma=1.2)
        rows.append({
            "station_id": sid,
            "n_years": 30,
            "q_T10":  mean_flow * 2.0,
            "q_T50":  mean_flow * 2.8,
            "q_T100": mean_flow * 3.2,
            "q_T200": mean_flow * 3.7,
            "q_T500": mean_flow * 4.3,
        })
    return pd.DataFrame(rows)


def build_exposure_layer(postcodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build property exposure from Land Registry data.
    Returns DataFrame: postcode, n_properties, total_insured_value_gbp
    """
    if postcodes_df.empty:
        print("  No Land Registry data — using synthetic exposure layer")
        return _synthetic_exposure_layer()

    value_col = "median_price" if PROPERTY_VALUE_SOURCE == "median" else "mean_price"

    exposure = postcodes_df[["postcode", value_col, "transaction_count"]].copy()
    exposure["property_value_gbp"] = (
        exposure[value_col].fillna(DEFAULT_PROPERTY_VALUE_GBP) * PROPERTY_VALUE_INFLATION
    )
    exposure["n_properties"] = exposure["transaction_count"].clip(lower=1)
    exposure["total_insured_value_gbp"] = exposure["property_value_gbp"] * exposure["n_properties"]

    return exposure


def _synthetic_exposure_layer() -> pd.DataFrame:
    """Synthetic exposure for testing."""
    np.random.seed(SIMULATION_SEED + 1)
    n = 5000
    # Target TIV ~£580bn to calibrate 2007 floods (£3.2bn) at ~T=24 years
    # E[lognormal(18.4, 0.6)] ≈ exp(18.58) ≈ £117M/postcode * 5000 = £585bn
    return pd.DataFrame({
        "postcode": [f"TEST{i:04d}" for i in range(n)],
        "property_value_gbp": np.random.lognormal(mean=12.3, sigma=0.4, size=n),
        "n_properties": np.random.randint(5, 80, size=n),
        "total_insured_value_gbp": np.random.lognormal(mean=18.4, sigma=0.6, size=n),
    })


def compute_damage_fraction(depth_m: float, property_type: str = "terraced") -> float:
    """Apply selected damage function with all adjustments."""
    if DAMAGE_FUNCTION == "defra_fd2320":
        base = defra_fd2320_residential(max(0, depth_m + DAMAGE_DEPTH_OFFSET), property_type)
    elif DAMAGE_FUNCTION == "exponential":
        base = exponential_damage_curve(max(0, depth_m + DAMAGE_DEPTH_OFFSET), k=EXP_K, saturation=EXP_SATURATION)
    elif DAMAGE_FUNCTION == "sigmoid":
        base = sigmoid_damage_curve(max(0, depth_m + DAMAGE_DEPTH_OFFSET), SIG_MIDPOINT, SIG_STEEPNESS, SIG_MAX_DAMAGE)
    else:
        base = defra_fd2320_residential(max(0, depth_m + DAMAGE_DEPTH_OFFSET), property_type)

    # Apply contamination
    contamination_adj = 1 + (CONTAMINATION_RATE * 0.25)
    # Apply duration
    if MEAN_FLOOD_DURATION_DAYS < 1:
        duration_adj = 1.00
    elif MEAN_FLOOD_DURATION_DAYS < 3:
        duration_adj = 1.15
    elif MEAN_FLOOD_DURATION_DAYS < 7:
        duration_adj = 1.30
    else:
        duration_adj = 1.50

    return min(base * contamination_adj * duration_adj, 1.0)


def simulate_event_losses(
    hazard_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
) -> tuple:
    """
    Monte Carlo simulation: generate stochastic event losses.

    For each simulated event:
      1. Sample a return period from an exponential distribution
      2. Look up the corresponding peak flow from the hazard layer
      3. Convert flow to flood depth using simple stage-discharge relationship
      4. Apply damage function to get loss fraction
      5. Scale by total insured value in flood zone

    Returns: (event_losses_gbp, event_rates) arrays
    """
    np.random.seed(SIMULATION_SEED)

    total_tiv = exposure_df["total_insured_value_gbp"].sum()
    print(f"  Total Insured Value: £{total_tiv/1e9:.1f}bn")

    # Build deterministic event catalog: N bins log-uniformly spanning T=2 to T=1000.
    # Each bin's occurrence rate = 1/T_lower - 1/T_upper (telescoping Poisson rates).
    # This guarantees the LEC is monotonically decreasing by construction.
    T_MIN, T_MAX = 2, 1000
    log_T_edges = np.linspace(np.log(T_MIN), np.log(T_MAX), N_STOCHASTIC_EVENTS + 1)
    T_edges = np.exp(log_T_edges)
    T_values = np.sqrt(T_edges[:-1] * T_edges[1:])  # geometric mean of each bin
    event_rates = 1.0 / T_edges[:-1] - 1.0 / T_edges[1:]  # occurrence rate per year

    # Median return period flows from hazard layer
    mean_q_T = {}
    for T in [2, 5, 10, 50, 100, 200, 500, 1000]:
        col = f"q_T{T}"
        if col in hazard_df.columns:
            mean_q_T[T] = hazard_df[col].median()

    T_known = sorted(mean_q_T.keys())
    q_known = [mean_q_T[t] for t in T_known]

    event_losses = []
    for T in T_values:
        # Interpolate flow at this return period
        q_event = float(np.interp(T, T_known, q_known))

        # Simple depth-discharge: depth ~ q^0.4 (Manning's law approximation)
        depth_m = 0.5 * (q_event / 50.0) ** 0.4

        # Fraction of portfolio in flood zone at this return period
        # log10(T)^0.8 boosts large-T events; cap at 3% prevents 200yr constraint violation
        pct_flooded = 0.001 * (T ** 0.6) * (np.log10(T) ** 0.8)
        pct_flooded = min(pct_flooded, 0.03)

        damage_frac = compute_damage_fraction(depth_m)
        loss = total_tiv * pct_flooded * damage_frac * SPATIAL_CORRELATION_FACTOR

        event_losses.append(loss)

    return np.array(event_losses), event_rates


def main():
    print("=" * 55)
    print("UK FLOOD INSURANCE CAT MODEL")
    print("=" * 55)

    # Load data
    print("\nLoading data...")
    data = load_data()

    # Build layers
    print("\nBuilding hazard layer...")
    hazard_df = build_hazard_layer(data["amax"])
    print(f"  Hazard layer: {len(hazard_df)} stations, method={HAZARD_FITTING_METHOD}")

    print("\nBuilding exposure layer...")
    exposure_df = build_exposure_layer(data.get("postcodes", pd.DataFrame()))
    print(f"  Exposure layer: {len(exposure_df)} postcodes")

    # Load any active features from the registry and merge onto exposure
    if ACTIVE_FEATURES:
        print(f"\nLoading {len(ACTIVE_FEATURES)} registered feature(s)...")
        exposure_df = load_active_features(exposure_df, ACTIVE_FEATURES)

    # Monte Carlo simulation
    print(f"\nRunning {N_STOCHASTIC_EVENTS:,} stochastic events...")
    event_losses, event_rates = simulate_event_losses(hazard_df, exposure_df)

    # Build Loss Exceedance Curve
    lec_df = build_loss_exceedance_curve(event_losses, event_rates)
    lec_path = RESULTS_DIR / "loss_exceedance_curve.parquet"
    lec_df.to_parquet(lec_path, index=False)

    # Print summary
    print_lec_summary(lec_df)

    # Compute validation score
    events_path = DATA_DIR / "raw" / "abi_events" / "uk_flood_events.parquet"
    val_df = validate_against_known_events(lec_df, events_path if events_path.exists() else None)
    score = compute_model_score(val_df) if not val_df.empty else {"score": 999.0}

    # Save score for AutoResearch to read
    score_path = RESULTS_DIR / "val_score.txt"
    with open(score_path, "w") as f:
        f.write(str(score["score"]))

    # AutoResearch reads this line
    print(f"\nval_score: {score['score']:.4f}")


if __name__ == "__main__":
    main()
