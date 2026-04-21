"""Trend state engine: classifies features as rising, falling, seasonal, or archived."""

from __future__ import annotations

from enum import Enum

import pandas as pd


class TrendState(str, Enum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    SEASONAL = "seasonal"
    EMERGING = "emerging"
    ARCHIVED = "archived"


def compute_trend_states(
    yearly_metrics: pd.DataFrame,
    rising_threshold: float = 0.15,
    falling_threshold: float = -0.15,
    min_years_for_seasonal: int = 2,
) -> pd.DataFrame:
    """Compute trend states from yearly feature counts.

    Args:
        yearly_metrics: DataFrame with columns [feature, year, count, share].
        rising_threshold: YoY share change above which a feature is "rising".
        falling_threshold: YoY share change below which a feature is "falling".
        min_years_for_seasonal: Minimum years a feature must appear to detect seasonality.

    Returns:
        DataFrame with columns [feature, year, count, share, yoy_delta, state].
    """
    df = yearly_metrics.copy().sort_values(["feature", "year"])
    df["yoy_delta"] = df.groupby("feature")["share"].diff()

    states: list[str] = []
    for _, row in df.iterrows():
        states.append(_classify_row(row, rising_threshold, falling_threshold))

    df["state"] = states
    return df


def _classify_row(row: pd.Series, rising_th: float, falling_th: float) -> str:
    delta = row.get("yoy_delta")

    if pd.isna(delta):
        return TrendState.EMERGING.value

    if delta > rising_th:
        return TrendState.RISING.value
    if delta < falling_th:
        return TrendState.FALLING.value

    share = row.get("share", 0)
    if share < 0.01:
        return TrendState.ARCHIVED.value

    return TrendState.STABLE.value


def detect_seasonal_features(
    yearly_metrics: pd.DataFrame,
    min_years: int = 2,
    variance_threshold: float = 0.05,
) -> set[str]:
    """Identify features that appear cyclically (e.g., linen peaks in summer)."""
    features_by_year = yearly_metrics.groupby("feature")["share"].agg(["count", "std", "mean"])
    seasonal = features_by_year[
        (features_by_year["count"] >= min_years) & (features_by_year["std"] > variance_threshold)
    ]
    return set(seasonal.index)


def get_top_features(
    yearly_metrics: pd.DataFrame,
    year: int,
    n: int = 20,
    category: str | None = None,
) -> pd.DataFrame:
    """Return top N features for a given year, optionally filtered by category."""
    mask = yearly_metrics["year"] == year
    if category:
        mask &= yearly_metrics["category"] == category
    return yearly_metrics[mask].nlargest(n, "share")
