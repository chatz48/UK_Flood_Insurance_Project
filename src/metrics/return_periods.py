"""
Loss Exceedance Curve and Return Period Loss Calculator

This is the PRIMARY OUTPUT of the cat model — the single most important thing
an insurer, reinsurer or regulator wants to see.

The Loss Exceedance Curve (LEC) shows:
  P(Annual Loss > X) for all X

Key points on the curve:
  T=10:   1-in-10 year loss    — used for pricing flood excess-of-loss reinsurance
  T=100:  1-in-100 year loss   — Solvency II internal model benchmark
  T=200:  1-in-200 year loss   — Solvency II SCR (Solvency Capital Requirement) standard
  T=500:  1-in-500 year loss   — Lloyd's Realistic Disaster Scenarios (RDS)

Annual Average Loss (AAL):
  The expected loss per year, averaging across all possible event frequencies.
  This is what an insurer adds to their base premium for flood cover.
  AAL = integral of the LEC from 0 to infinity
      = sum(loss_i * probability_i) across all simulated events

Validation:
  We validate by checking our modelled event losses against the ~13 known
  UK flood events since 2000 with published ABI/DEFRA insured loss estimates.
"""

import numpy as np
import pandas as pd
from scipy import integrate
from pathlib import Path


RETURN_PERIODS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def build_loss_exceedance_curve(
    event_losses_gbp: np.ndarray,
    event_rates: np.ndarray,
) -> pd.DataFrame:
    """
    Build the Loss Exceedance Curve from a set of simulated events.

    Args:
        event_losses_gbp: array of losses for each simulated event (£)
        event_rates: annual frequency (occurrence rate) for each event
                     e.g. a 1-in-100 year event has rate 0.01

    Returns:
        DataFrame with columns: loss_gbp, exceedance_prob, return_period_years
        Sorted by descending loss.
    """
    # Sort events from largest to smallest loss
    order = np.argsort(event_losses_gbp)[::-1]
    losses = event_losses_gbp[order]
    rates = event_rates[order]

    # Exceedance rate = cumulative sum of rates of events larger than this loss
    # (assumes events are independent Poisson processes)
    cumulative_rates = np.cumsum(rates)

    # Convert annual rate to annual exceedance probability
    # P(exceed) = 1 - exp(-rate) for Poisson process
    exceedance_probs = 1 - np.exp(-cumulative_rates)
    return_periods = 1 / cumulative_rates

    return pd.DataFrame({
        "loss_gbp": losses,
        "exceedance_rate": cumulative_rates,
        "exceedance_prob": exceedance_probs,
        "return_period_years": return_periods,
    })


def compute_aal(lec_df: pd.DataFrame) -> float:
    """
    Compute Annual Average Loss (AAL) by integrating the LEC.

    AAL = integral from 0 to infinity of P(L > x) dx
        = sum_i(loss_i * rate_i) for discrete event set

    This is the expected value of annual losses.
    """
    losses = lec_df["loss_gbp"].values
    rates = lec_df["exceedance_rate"].values

    # Numerical integration using the trapezoidal rule on the LEC
    # Sort by ascending loss for integration
    order = np.argsort(losses)
    losses_sorted = losses[order]
    probs_sorted = lec_df["exceedance_prob"].values[order]

    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    aal = trapz(probs_sorted[::-1], losses_sorted)
    return float(aal)


def get_return_period_losses(lec_df: pd.DataFrame, return_periods: list = None) -> dict:
    """
    Interpolate losses at standard return periods from the LEC.

    Returns: {T: loss_gbp} dict for requested return periods.
    """
    if return_periods is None:
        return_periods = RETURN_PERIODS

    results = {}
    for T in return_periods:
        target_rate = 1 / T
        # Interpolate loss at this exceedance rate
        loss = np.interp(
            target_rate,
            lec_df["exceedance_rate"].values[::-1],  # must be ascending
            lec_df["loss_gbp"].values[::-1],
        )
        results[T] = float(loss)

    return results


def compute_tail_value_at_risk(lec_df: pd.DataFrame, percentile: float = 0.995) -> float:
    """
    Compute Tail Value at Risk (TVaR) at a given confidence level.
    Also called Conditional Tail Expectation (CTE) or Expected Shortfall.

    TVaR(99.5%) = average loss given that a 1-in-200 year event has occurred.
    This is what Solvency II requires insurers to hold capital for.

    percentile: 0.995 = 99.5th percentile (1-in-200 year)
    """
    threshold_prob = 1 - percentile
    tail = lec_df[lec_df["exceedance_prob"] >= threshold_prob]
    if tail.empty:
        return float(lec_df["loss_gbp"].max())
    return float(tail["loss_gbp"].mean())


def validate_against_known_events(
    lec_df: pd.DataFrame,
    known_events_path: Path = None,
) -> pd.DataFrame:
    """
    Validate the model by comparing return period losses to known UK flood events.

    For each known event:
      1. Find where its insured loss falls on our LEC
      2. Compare implied return period to our modelled return period
      3. If 2007 floods (£3.2bn, ~1-in-75 year event) maps to T=50-100: good calibration

    Returns validation summary DataFrame.
    """
    if known_events_path is None:
        known_events_path = Path(__file__).parents[2] / "data" / "raw" / "abi_events" / "uk_flood_events.parquet"

    if not known_events_path.exists():
        print("Known events not found. Run abi_events.py pipeline first.")
        return pd.DataFrame()

    events = pd.read_parquet(known_events_path)
    events = events[events["confidence"].isin(["high", "medium"])]

    validation_rows = []
    for _, event in events.iterrows():
        insured_loss = event["insured_loss_gbp_m"] * 1_000_000  # convert to £

        # Find implied return period from our LEC
        implied_T = np.interp(
            insured_loss,
            lec_df["loss_gbp"].values[::-1],
            lec_df["return_period_years"].values[::-1],
        )

        validation_rows.append({
            "event": event["name"],
            "year": event["year"],
            "observed_loss_gbpm": event["insured_loss_gbp_m"],
            "implied_return_period_modelled": round(implied_T, 1),
            "confidence": event["confidence"],
        })

    val_df = pd.DataFrame(validation_rows).sort_values("observed_loss_gbpm", ascending=False)
    return val_df


def compute_model_score(validation_df: pd.DataFrame) -> dict:
    """
    Single composite score for model quality — used by AutoResearch as objective metric.

    Computes:
      - Log-RMSE on implied return periods (penalises large relative errors)
      - Bias (systematic over/under estimation)
      - Tail accuracy (how well we estimate the largest events)

    Lower score = better model.
    AutoResearch minimises this.
    """
    if validation_df.empty or "implied_return_period_modelled" not in validation_df.columns:
        return {"score": 999.0, "log_rmse": 999.0, "bias": 0.0}

    # True return periods (approximate — based on event history 2000-2024 = 24 years)
    # The 2007 event is approximately 1-in-75, the 2013/14 approximately 1-in-25, etc.
    # We use the rank-based empirical return periods
    n_years = 24  # years of comparable record
    df = validation_df.copy()
    df = df.sort_values("observed_loss_gbpm", ascending=False).reset_index(drop=True)
    df["empirical_T"] = n_years / (df.index + 1)

    log_errors = np.log(df["implied_return_period_modelled"] + 1) - np.log(df["empirical_T"] + 1)
    log_rmse = float(np.sqrt(np.mean(log_errors**2)))
    bias = float(np.mean(log_errors))

    # Extra weight on events with loss > £1bn (tail accuracy matters most)
    tail_mask = df["observed_loss_gbpm"] >= 1000
    tail_log_rmse = float(np.sqrt(np.mean(log_errors[tail_mask]**2))) if tail_mask.any() else log_rmse

    # Composite score (lower is better)
    score = log_rmse + 0.5 * abs(bias) + 0.3 * tail_log_rmse

    return {
        "score": round(score, 4),
        "log_rmse": round(log_rmse, 4),
        "bias": round(bias, 4),
        "tail_log_rmse": round(tail_log_rmse, 4),
        "n_events_validated": len(df),
    }


def print_lec_summary(lec_df: pd.DataFrame, known_events_path: Path = None):
    """Print a human-readable summary of the Loss Exceedance Curve."""
    rp_losses = get_return_period_losses(lec_df)
    aal = compute_aal(lec_df)
    tvar = compute_tail_value_at_risk(lec_df)

    print("\n" + "="*55)
    print("LOSS EXCEEDANCE CURVE SUMMARY")
    print("="*55)
    print(f"\nAnnual Average Loss (AAL):          £{aal/1e6:,.0f}m")
    print(f"TVaR (99.5% / 1-in-200 year):       £{tvar/1e6:,.0f}m")
    print("\nReturn Period Losses:")
    for T, loss in rp_losses.items():
        print(f"  1-in-{T:<5} year:  £{loss/1e6:>8,.0f}m")

    val_df = validate_against_known_events(lec_df, known_events_path)
    if not val_df.empty:
        print("\nValidation against known UK flood events:")
        print(val_df.to_string(index=False))
        score = compute_model_score(val_df)
        print(f"\nModel score (lower=better): {score['score']:.4f}")
        print(f"  Log-RMSE:       {score['log_rmse']:.4f}")
        print(f"  Bias:           {score['bias']:.4f}")
        print(f"  Tail accuracy:  {score['tail_log_rmse']:.4f}")
    print("="*55)
