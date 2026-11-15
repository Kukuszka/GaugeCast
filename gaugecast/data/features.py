from __future__ import annotations

import numpy as np
import pandas as pd


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map column names used by the notebook experiments to GaugeCast canonical names."""
    df = df.copy()
    aliases = {
        "datetime": "dates",
        "datetime_london": "dates",
        "value": "level",
        "rain": "rain_edge_green",
        "forecast_rain": "future_rain",
    }
    existing = {k: v for k, v in aliases.items() if k in df.columns and v not in df.columns}
    if existing:
        df = df.rename(columns=existing)
    if "dates" in df.columns:
        df["dates"] = pd.to_datetime(df["dates"])
    return df


def engineer_features(df, use_temp_vwc=True):
    df = df.copy().sort_values("dates").reset_index(drop=True)

    if "temp05" not in df.columns:
        df["temp05"] = 0.0
    if "vwc05" not in df.columns:
        df["vwc05"] = 0.0
    if "future_rain" not in df.columns:
        raise KeyError("future_rain column is missing.")

    numeric_cols = ["level", "rain_edge_green", "future_rain", "temp05", "vwc05"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # log-transform rainfall
    df["rain_edge_green"] = np.log1p(np.clip(df["rain_edge_green"], 0, None))
    df["future_rain"] = np.log1p(np.clip(df["future_rain"], 0, None))

    # level dynamics
    df["dh"] = df["level"].diff().fillna(0.0)
    df["d2h"] = df["dh"].diff().fillna(0.0)

    # antecedent wetness index
    decay_hours = 24 * 5
    alpha = np.exp(-1.0 / decay_hours)
    awi = [0.0]
    for i in range(1, len(df)):
        awi.append(alpha * awi[-1] + df.loc[i - 1, "rain_edge_green"])
    df["AWI"] = awi

    # rolling rain memory
    df["rain_3h"] = df["rain_edge_green"].rolling(window=12, min_periods=1).sum()
    df["rain_24h"] = df["rain_edge_green"].rolling(window=96, min_periods=1).sum()

    if use_temp_vwc:
        feature_cols_local = [
            "level",
            "rain_edge_green",
            "dh",
            "d2h",
            "future_rain",
            "AWI",
            "rain_3h",
            "rain_24h",
            "temp05",
            "vwc05"
        ]
    else:
        feature_cols_local = [
            "level",
            "rain_edge_green",
            "dh",
            "d2h",
            "future_rain",
            "AWI",
            "rain_3h",
            "rain_24h"
        ]

    df = df.dropna(subset=feature_cols_local).reset_index(drop=True)
    return df, feature_cols_local
