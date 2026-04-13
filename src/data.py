"""Truflation data loading and preprocessing for Kairos experiments."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data" / "truflation"

# Truflation CPI hierarchy: column name prefix → category
TRUFLATION_CATEGORIES = [
    "Food & Non-alcoholic Beverages",
    "Food at home",
    "Cereals",
    "Meats",
    "Dairy",
    "Fruits",
    "Other foods at home",
    "Food away from home",
    "Housing",
    "Owned dwellings",
    "Rented dwellings",
    "Other lodging",
    "Transport",
    "Vehicle purchases (net outlay)",
    "Gasoline, other fuels, and motor oil",
    "Public and other transportation",
    "Utilities",
    "Natural gas",
    "Electricity",
    "Health",
    "Household Durables & Daily Use Items",
    "Housekeeping supplies",
    "Household furnishings and equipment",
    "Alcohol & Tobacco",
    "Alcoholic beverages",
    "Tobacco Products and Smoking Supplies",
    "Clothing & Footwear",
    "Women and girls",
    "Communications",
    "Education",
    "Recreation & Culture",
    "Other",
]

AGGREGATE_CATEGORIES = ["Goods", "Services", "Core", "NonCore"]


def load_cpi(frozen: bool = True) -> pd.DataFrame:
    """Load headline CPI data.

    Args:
        frozen: If True, load frozen (point-in-time) data. If False, load
                unfrozen (revised) data.

    Returns:
        DataFrame with columns: date, inflation (YoY), cpiIndex, cpiIndexYearAgo
    """
    subdir = "frozen" if frozen else "unfrozen"
    path = DATA_DIR / subdir / "us_cpi.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.drop(columns=["created_at"], errors="ignore")
    return df.sort_values("date").reset_index(drop=True)


def load_categories(frozen: bool = True) -> pd.DataFrame:
    """Load full category-level data (383 columns).

    Args:
        frozen: If True, load frozen data.

    Returns:
        DataFrame with date index and all category columns.
    """
    subdir = "frozen" if frozen else "unfrozen"
    path = DATA_DIR / subdir / "us_categories.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.drop(columns=["created_at"], errors="ignore")
    return df.sort_values("date").reset_index(drop=True)


def extract_index_series(
    df: pd.DataFrame,
    categories: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Extract Index columns (not YoY or YearAgo) for specified categories.

    These are the raw index values (base 100 at Jan 2010) — the primary
    training signal for forecasting.

    Args:
        df: Full categories DataFrame from load_categories().
        categories: List of category names. Defaults to all Truflation categories.

    Returns:
        DataFrame with date + one column per category (index values).
    """
    if categories is None:
        categories = TRUFLATION_CATEGORIES

    cols = ["date"]
    for cat in categories:
        index_col = f"{cat}Index"
        if index_col in df.columns:
            cols.append(index_col)

    result = df[cols].copy()
    result.columns = ["date"] + [
        c.replace("Index", "") for c in cols[1:]
    ]
    return result


def extract_yoy_series(
    df: pd.DataFrame,
    categories: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Extract YoY (year-over-year change) columns for specified categories.

    Args:
        df: Full categories DataFrame.
        categories: List of category names. Defaults to all Truflation categories.

    Returns:
        DataFrame with date + one column per category (YoY values).
    """
    if categories is None:
        categories = TRUFLATION_CATEGORIES

    cols = ["date"]
    for cat in categories:
        yoy_col = f"{cat}YoY"
        if yoy_col in df.columns:
            cols.append(yoy_col)

    result = df[cols].copy()
    result.columns = ["date"] + [
        c.replace("YoY", "") for c in cols[1:]
    ]
    return result


def extract_bls_official(df: pd.DataFrame) -> pd.DataFrame:
    """Extract official BLS CPI data (ground truth for evaluation).

    Returns:
        DataFrame with date + BLS sub-category index values and YoY.
    """
    bls_cols = ["date"] + [c for c in df.columns if c.startswith("BLS ")]
    return df[bls_cols].copy()


def extract_bea_pce(df: pd.DataFrame) -> pd.DataFrame:
    """Extract official BEA PCE data (ground truth for evaluation).

    Returns:
        DataFrame with date + BEA PCE sub-category values.
    """
    bea_cols = ["date"] + [c for c in df.columns if c.startswith("BEA PCE")]
    return df[bea_cols].copy()


def train_test_split_temporal(
    df: pd.DataFrame,
    test_start: str = "2024-01-01",
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by date — never leak future into training.

    Args:
        df: DataFrame with a date column.
        test_start: First date of test set (YYYY-MM-DD).
        date_col: Name of the date column.

    Returns:
        (train_df, test_df) tuple.
    """
    cutoff = pd.Timestamp(test_start)
    train = df[df[date_col] < cutoff].copy()
    test = df[df[date_col] >= cutoff].copy()
    return train, test


def prepare_series_for_model(
    df: pd.DataFrame,
    date_col: str = "date",
) -> list[dict]:
    """Convert DataFrame to list of univariate series dicts for TSFM evaluation.

    Each series dict has:
        - name: column name
        - values: numpy array of float values
        - dates: numpy array of dates
        - freq: "D" (daily)

    Drops NaN-only series and forward-fills sparse NaNs.
    """
    series_list = []
    value_cols = [c for c in df.columns if c != date_col]
    dates = df[date_col].values

    for col in value_cols:
        values = df[col].values.astype(float)
        if np.all(np.isnan(values)):
            continue
        values = pd.Series(values).ffill().bfill().values
        series_list.append({
            "name": col,
            "values": values,
            "dates": dates,
            "freq": "D",
        })

    return series_list
