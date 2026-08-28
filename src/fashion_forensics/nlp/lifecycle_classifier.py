"""Classify each trend's temporal behavior from §3.7's monthly frequency table.

Implements ARCHITECTURE.md §3.8. Rule-based classifier over derived
time-series features (active-month run-length, peak, trailing slope,
recurrence after inactivity) — NOT probabilistic survival analysis, per
ARCHITECTURE.md's explicit caution.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

import pandas as pd

from fashion_forensics.config import PROJECT_ROOT

LIFECYCLE_VERSION = "v1"

LIFECYCLE_STATES_PATH = (
    PROJECT_ROOT / "data" / "03_shared" / "trend_lifecycle" / "lifecycle_states.jsonl"
)


class LifecycleState(StrEnum):
    RISING = "rising"
    PERSISTENT = "persistent"
    SEASONAL_RECURRING = "seasonal_recurring"
    DECLINING = "declining"
    INACTIVE = "inactive"


def _merged_active_runs(
    dates_active: list[tuple[str, bool]], recurrence_gap_months: int
) -> list[list[int]]:
    """Given [(date, is_active), ...] sorted chronologically, return the
    index-runs of active months, merging two active runs separated by a
    gap shorter than `recurrence_gap_months` into a single run (treats a
    short blip of inactivity as noise, not a real deactivation).
    """
    active_idx = [i for i, (_, active) in enumerate(dates_active) if active]
    if not active_idx:
        return []

    runs: list[list[int]] = [[active_idx[0]]]
    for idx in active_idx[1:]:
        gap = idx - runs[-1][-1]
        if gap <= recurrence_gap_months:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return runs


def compute_lifecycle_features(
    monthly_frequency: pd.DataFrame,
    trend_field: str = "trend",
    date_field: str = "date",
    exclude_labels: tuple[str, ...] = ("unknown",),
    recurrence_gap_months: int = 2,
    slope_window: int = 3,
) -> pd.DataFrame:
    """Pure feature extraction, no state assigned.

    Args:
        monthly_frequency: output of
            time_series_aggregation.aggregate_monthly_frequency (or any
            DataFrame with the same [date, trend, trend_frequency,
            is_active] shape). Must have one row per (date, trend) with no
            gaps in the date sequence per trend — the run-length logic
            depends on this zero-fill guarantee.
        trend_field / date_field: column names.
        exclude_labels: labels dropped before analysis (e.g. "unknown",
            which isn't zero-filled by §3.7 and so lacks the one-row-per-
            month guarantee the run-length logic needs).
        recurrence_gap_months: active runs separated by a gap this short
            (in months) are merged into one run rather than counted as
            separate active periods.
        slope_window: number of trailing points (within the last active
            run) used to compute recent_slope.

    Returns:
        DataFrame, one row per trend: [trend, first_observed, peak_month,
        last_active, active_month_count, active_run_length,
        num_active_periods, months_since_last_active, recent_slope].
        first_observed/last_active/peak_month are None for a trend that's
        never active (or, for peak_month, never has any trend_count > 0).
    """
    df = monthly_frequency[~monthly_frequency[trend_field].isin(exclude_labels)].copy()
    all_dates = sorted(df[date_field].unique())
    global_latest_idx = len(all_dates) - 1

    rows: list[dict] = []
    for trend, group in df.groupby(trend_field):
        group = group.set_index(date_field).reindex(all_dates)
        group["is_active"] = group["is_active"].fillna(False)
        dates_active = list(zip(all_dates, group["is_active"].tolist(), strict=True))

        active_dates = [d for d, a in dates_active if a]
        first_observed = active_dates[0] if active_dates else None
        last_active = active_dates[-1] if active_dates else None

        freq_series = group["trend_frequency"]
        if freq_series.notna().any() and freq_series.max() > 0:
            peak_month = freq_series.idxmax()
        else:
            peak_month = None

        runs = _merged_active_runs(dates_active, recurrence_gap_months)
        active_month_count = len(active_dates)
        num_active_periods = len(runs)

        if runs:
            last_run = runs[-1]
            active_run_length = len(last_run)
            last_active_idx = last_run[-1]
            months_since_last_active = global_latest_idx - last_active_idx

            window_idx = last_run[-slope_window:]
            if len(window_idx) >= 2:
                freqs = [freq_series.iloc[i] for i in window_idx]
                recent_slope = (freqs[-1] - freqs[0]) / (len(freqs) - 1)
            else:
                recent_slope = 0.0
        else:
            active_run_length = 0
            months_since_last_active = None
            recent_slope = 0.0

        rows.append(
            {
                "trend": trend,
                "first_observed": first_observed,
                "peak_month": peak_month,
                "last_active": last_active,
                "active_month_count": active_month_count,
                "active_run_length": active_run_length,
                "num_active_periods": num_active_periods,
                "months_since_last_active": months_since_last_active,
                "recent_slope": recent_slope,
            }
        )

    return pd.DataFrame(rows).sort_values("trend").reset_index(drop=True)


def classify_lifecycles(
    monthly_frequency: pd.DataFrame,
    trend_field: str = "trend",
    date_field: str = "date",
    persistent_min_active_months: int = 4,
    rising_slope_threshold: float = 0.03,
    declining_slope_threshold: float = -0.03,
    slope_window: int = 3,
    recency_inactive_months: int = 2,
    recurrence_gap_months: int = 2,
    exclude_labels: tuple[str, ...] = ("unknown",),
    lifecycle_version: str = LIFECYCLE_VERSION,
) -> pd.DataFrame:
    """Classify each trend's lifecycle state from its monthly frequency history.

    Thresholds are explicit, parameterized first-pass defaults (same spirit
    as trend_engine.py's rising_threshold=0.15) — not tuned against a real
    distribution, since only a handful of canonical trends exist today.

    Decision procedure (first match wins, per trend):
      1. active_month_count == 0                          -> INACTIVE
      2. not currently active (months_since_last_active
         > recency_inactive_months):
           num_active_periods >= 2                         -> SEASONAL_RECURRING
           else                                             -> INACTIVE
      3. currently active:
           recent_slope <= declining_slope_threshold        -> DECLINING
           active_run_length >= persistent_min_active_months
             and abs(recent_slope) < rising_slope_threshold -> PERSISTENT
           recent_slope >= rising_slope_threshold
             or active_run_length < persistent_min_active_months -> RISING
           fallback                                         -> PERSISTENT

    Returns:
        compute_lifecycle_features' columns plus [state, lifecycle_version].
    """
    features = compute_lifecycle_features(
        monthly_frequency,
        trend_field=trend_field,
        date_field=date_field,
        exclude_labels=exclude_labels,
        recurrence_gap_months=recurrence_gap_months,
        slope_window=slope_window,
    )

    states: list[str] = []
    for _, row in features.iterrows():
        if row["active_month_count"] == 0:
            states.append(LifecycleState.INACTIVE.value)
            continue

        if row["months_since_last_active"] > recency_inactive_months:
            if row["num_active_periods"] >= 2:
                states.append(LifecycleState.SEASONAL_RECURRING.value)
            else:
                states.append(LifecycleState.INACTIVE.value)
            continue

        slope = row["recent_slope"]
        if slope <= declining_slope_threshold:
            states.append(LifecycleState.DECLINING.value)
        elif (
            row["active_run_length"] >= persistent_min_active_months
            and abs(slope) < rising_slope_threshold
        ):
            states.append(LifecycleState.PERSISTENT.value)
        elif (
            slope >= rising_slope_threshold
            or row["active_run_length"] < persistent_min_active_months
        ):
            states.append(LifecycleState.RISING.value)
        else:
            states.append(LifecycleState.PERSISTENT.value)

    features["state"] = states
    features["lifecycle_version"] = lifecycle_version
    return features


def load_lifecycle_states(path: Path | None = None) -> dict[str, dict]:
    """Load classify_lifecycles' output (lifecycle_states.jsonl), keyed by trend.

    Raises FileNotFoundError with a message pointing at
    scripts/compute_trend_lifecycle.py if the file doesn't exist yet.
    """
    target = LIFECYCLE_STATES_PATH if path is None else Path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"{target} does not exist. Run scripts/compute_trend_lifecycle.py first to produce it."
        )
    out: dict[str, dict] = {}
    for line in target.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["trend"]] = rec
    return out
