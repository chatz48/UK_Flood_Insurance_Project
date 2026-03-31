"""
Flood Frequency Analysis
Fits extreme value distributions to NRFA AMAX/POT data to produce return period curves.

This is the hazard layer: given a gauging station, what flow rate corresponds to
the 1-in-10, 1-in-100, 1-in-200 year event?

Distributions fitted:
  GEV  — Generalised Extreme Value (fits AMAX directly, FEH standard)
  GPD  — Generalised Pareto Distribution (fits POT exceedances)
  GUM  — Gumbel (special case of GEV with shape=0, historically common)
  LP3  — Log-Pearson Type 3 (US standard, for comparison)

Fitting methods:
  L-moments: robust to outliers, standard in FEH
  MLE: maximum likelihood, better for longer records
  PWM: probability weighted moments (equivalent to L-moments)

Output: for each station, return period flows at T = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
"""

import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
from typing import Optional

RETURN_PERIODS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


# ===========================================================================
# L-Moments (FEH standard fitting method for flood frequency)
# ===========================================================================

def lmom_ratios(data: np.ndarray) -> tuple:
    """
    Compute L-moment ratios (l1, l2, t3, t4) from a sample.
    These are the inputs for fitting GEV via the FEH method.

    l1 = L-mean (location)
    l2 = L-scale (spread, always positive)
    t3 = L-skewness (shape of distribution)
    t4 = L-kurtosis
    """
    x = np.sort(data)
    n = len(x)
    if n < 4:
        return None

    # Probability weighted moments
    b0 = np.mean(x)
    b1 = sum((i / (n - 1)) * x[i] for i in range(n)) / n
    b2 = sum((i * (i - 1) / ((n - 1) * (n - 2))) * x[i] for i in range(n)) / n
    b3 = sum((i * (i - 1) * (i - 2) / ((n - 1) * (n - 2) * (n - 3))) * x[i] for i in range(n)) / n

    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0

    t3 = l3 / l2 if l2 > 0 else 0  # L-skewness
    t4 = l4 / l2 if l2 > 0 else 0  # L-kurtosis

    return l1, l2, t3, t4


def fit_gev_lmom(data: np.ndarray) -> dict:
    """
    Fit GEV distribution using L-moments (FEH method).
    Returns location (mu), scale (sigma), shape (xi) parameters.

    GEV CDF: F(x) = exp(-[1 + xi*(x-mu)/sigma]^(-1/xi))
    xi > 0: heavy tail (Frechet) — typical for flood data
    xi = 0: Gumbel (thin tail)
    xi < 0: bounded upper tail (Weibull)
    """
    lmom = lmom_ratios(data)
    if lmom is None:
        return None

    l1, l2, t3, _ = lmom

    # Rational approximation for shape parameter from L-skewness
    # From Hosking & Wallis (1997), valid for -0.5 < t3 < 0.5
    c = 2 / (3 + t3) - np.log(2) / np.log(3)
    xi = 7.8590 * c + 2.9554 * c**2

    # Scale and location from L-moments
    gamma_val = math.gamma(1 + xi) if abs(xi) > 1e-6 else 1.0
    sigma = l2 * xi / ((1 - 2**(-xi)) * gamma_val)
    mu = l1 - sigma * (gamma_val - 1) / xi if abs(xi) > 1e-6 else l1 - 0.5772 * sigma

    return {"mu": mu, "sigma": sigma, "xi": xi, "method": "lmom", "dist": "GEV"}


def fit_gev_mle(data: np.ndarray) -> dict:
    """Fit GEV using Maximum Likelihood Estimation."""
    try:
        xi, mu, sigma = stats.genextreme.fit(data)
        # scipy uses negated shape convention
        return {"mu": mu, "sigma": sigma, "xi": -xi, "method": "mle", "dist": "GEV"}
    except Exception:
        return None


def fit_gpd_mle(exceedances: np.ndarray) -> dict:
    """
    Fit Generalised Pareto Distribution to POT exceedances.
    exceedances = values above the threshold, minus the threshold.
    """
    try:
        xi, loc, sigma = stats.genpareto.fit(exceedances, floc=0)
        return {"sigma": sigma, "xi": xi, "method": "mle", "dist": "GPD"}
    except Exception:
        return None


def fit_gumbel_lmom(data: np.ndarray) -> dict:
    """Fit Gumbel (EV1) distribution using L-moments."""
    lmom = lmom_ratios(data)
    if lmom is None:
        return None
    l1, l2, _, _ = lmom
    sigma = l2 / np.log(2)
    mu = l1 - 0.5772 * sigma
    return {"mu": mu, "sigma": sigma, "xi": 0.0, "method": "lmom", "dist": "Gumbel"}


# ===========================================================================
# Return period computation
# ===========================================================================

def gev_quantile(T: float, mu: float, sigma: float, xi: float) -> float:
    """
    Compute the T-year return period flow from GEV parameters.
    T-year event = flow exceeded with probability 1/T in any year.
    """
    p = 1 - 1 / T  # non-exceedance probability
    if abs(xi) < 1e-6:
        # Gumbel special case
        return mu - sigma * np.log(-np.log(p))
    else:
        return mu + sigma * ((-np.log(p))**(-xi) - 1) / xi


def compute_return_period_flows(params: dict, return_periods: list = None) -> dict:
    """
    Given fitted distribution parameters, compute flows at each return period.
    Returns dict: {T: flow_m3s}
    """
    if params is None:
        return {}
    if return_periods is None:
        return_periods = RETURN_PERIODS

    results = {}
    for T in return_periods:
        try:
            q = gev_quantile(T, params["mu"], params["sigma"], params["xi"])
            results[T] = round(q, 3)
        except Exception:
            results[T] = np.nan

    return results


# ===========================================================================
# Bootstrap confidence intervals
# ===========================================================================

def bootstrap_return_periods(
    data: np.ndarray,
    fit_func,
    n_bootstrap: int = 1000,
    return_periods: list = None,
) -> pd.DataFrame:
    """
    Bootstrap confidence intervals for return period estimates.
    Returns DataFrame with columns: T, median, ci_lower_90, ci_upper_90
    """
    if return_periods is None:
        return_periods = RETURN_PERIODS

    n = len(data)
    bootstrap_quantiles = {T: [] for T in return_periods}

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        params = fit_func(sample)
        if params is None:
            continue
        for T in return_periods:
            q = gev_quantile(T, params["mu"], params["sigma"], params["xi"])
            bootstrap_quantiles[T].append(q)

    results = []
    for T in return_periods:
        vals = np.array(bootstrap_quantiles[T])
        results.append({
            "T": T,
            "median": np.median(vals),
            "ci_lower_90": np.percentile(vals, 5),
            "ci_upper_90": np.percentile(vals, 95),
            "ci_lower_95": np.percentile(vals, 2.5),
            "ci_upper_95": np.percentile(vals, 97.5),
        })

    return pd.DataFrame(results)


# ===========================================================================
# Main analysis function
# ===========================================================================

def analyse_station(
    station_id: int,
    amax_data: pd.DataFrame,
    pot_data: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Full flood frequency analysis for one gauging station.

    Fits GEV (L-moments + MLE), Gumbel, and optionally GPD (POT).
    Returns return period flows under each method for comparison.
    """
    station_amax = amax_data[amax_data["station_id"] == station_id]["peak_flow_m3s"].dropna().values

    if len(station_amax) < 5:
        return {"station_id": station_id, "error": "insufficient_data", "n_years": len(station_amax)}

    results = {
        "station_id": station_id,
        "n_years": len(station_amax),
        "mean_flow": float(np.mean(station_amax)),
        "max_flow": float(np.max(station_amax)),
        "cv": float(np.std(station_amax) / np.mean(station_amax)),
    }

    # Fit distributions
    fits = {
        "gev_lmom": fit_gev_lmom(station_amax),
        "gev_mle": fit_gev_mle(station_amax),
        "gumbel_lmom": fit_gumbel_lmom(station_amax),
    }

    # Return period flows per method
    for method_name, params in fits.items():
        if params:
            rp_flows = compute_return_period_flows(params)
            results[method_name] = {
                "params": params,
                "return_periods": rp_flows,
            }

    # L-moment ratios (diagnostic)
    lmom = lmom_ratios(station_amax)
    if lmom:
        results["l_skewness"] = float(lmom[2])
        results["l_kurtosis"] = float(lmom[3])

    return results


def filter_stations_by_qmed(amax_df: pd.DataFrame, min_qmed: float = 5.0) -> pd.DataFrame:
    """
    Filter AMAX data to stations with QMED > min_qmed m³/s.

    QMED (Median Annual Maximum flow) is the 2-year return period flow —
    the most statistically robust single descriptor of a catchment's flood
    response. Filtering to QMED > 5 m³/s removes small Scottish/upland
    streams that distort the UK-wide hazard layer median.

    Previously (discarded experiment): using all 887 stations produced
    median q_T2 = 37.4 m³/s, far too low; this filter restores calibration.
    """
    qmed = (
        amax_df.groupby("station_id")["peak_flow_m3s"]
        .median()
        .reset_index()
        .rename(columns={"peak_flow_m3s": "qmed"})
    )
    valid = qmed[qmed["qmed"] > min_qmed]["station_id"]
    filtered = amax_df[amax_df["station_id"].isin(valid)]
    n_before = amax_df["station_id"].nunique()
    n_after = filtered["station_id"].nunique()
    print(f"  QMED filter (>{min_qmed} m³/s): {n_before} → {n_after} stations "
          f"({n_before - n_after} small streams removed)")
    return filtered


def run_full_analysis(amax_path: Path = None) -> pd.DataFrame:
    """
    Run flood frequency analysis on all NRFA stations.
    Outputs a DataFrame with return period flows per station per method.
    """
    if amax_path is None:
        amax_path = Path(__file__).parents[2] / "data" / "raw" / "nrfa" / "amax_all_stations.parquet"

    if not amax_path.exists():
        print(f"AMAX data not found at {amax_path}")
        print("Run src/pipelines/nrfa_peaks.py first.")
        return pd.DataFrame()

    amax_df = pd.read_parquet(amax_path)
    stations = amax_df["station_id"].unique()
    print(f"Running flood frequency analysis on {len(stations)} stations...")

    all_results = []
    for sid in stations:
        result = analyse_station(sid, amax_df)
        # Flatten return period flows to columns
        row = {
            "station_id": sid,
            "n_years": result.get("n_years", 0),
            "mean_flow": result.get("mean_flow"),
            "max_flow": result.get("max_flow"),
            "cv": result.get("cv"),
            "l_skewness": result.get("l_skewness"),
        }
        # Add return period flows for primary method (GEV L-moments)
        if "gev_lmom" in result:
            for T, q in result["gev_lmom"]["return_periods"].items():
                row[f"q_T{T}_gev_lmom"] = q

        if "gev_mle" in result:
            for T, q in result["gev_mle"]["return_periods"].items():
                row[f"q_T{T}_gev_mle"] = q

        all_results.append(row)

    results_df = pd.DataFrame(all_results)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "flood_frequency_results.parquet"
    results_df.to_parquet(out_path, index=False)
    print(f"Saved frequency analysis to {out_path}")

    return results_df


if __name__ == "__main__":
    run_full_analysis()
