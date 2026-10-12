"""
Data Loader
============
Loads and validates training and prediction data (CSV) for the
AI Threat Detection system.

Expected CSV columns:
  - timestamp    : ISO 8601 datetime
  - event_type   : e.g., "login_failed", "port_scan", "dns_query"
  - source_ip    : e.g., "192.168.1.10"
  - user         : e.g., "admin", "root"
  - port         : integer (or empty)
  - bytes        : integer (or empty)
  - label        : "malicious" / "benign"  (training only)
"""

import os
import pandas as pd


REQUIRED_TRAIN_COLUMNS = [
    "timestamp", "event_type", "source_ip",
    "user", "port", "bytes", "label",
]

REQUIRED_PREDICT_COLUMNS = [
    "timestamp", "event_type", "source_ip",
    "user", "port", "bytes",
]


def load_training_data(filepath):
    """Load and validate training CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Validate columns
    missing = set(REQUIRED_TRAIN_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in training data: {missing}")

    # Clean
    df = df.fillna({
        "user": "unknown",
        "port": 0,
        "bytes": 0,
    })
    df["user"] = df["user"].astype(str)
    df["source_ip"] = df["source_ip"].astype(str)
    df["event_type"] = df["event_type"].astype(str)
    df["port"] = pd.to_numeric(df["port"], errors="coerce").fillna(0).astype(int)
    df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce").fillna(0).astype(int)

    return df


def load_new_events(filepath):
    """Load and validate new events for prediction."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Events file not found: {filepath}")

    df = pd.read_csv(filepath)

    missing = set(REQUIRED_PREDICT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in events data: {missing}")

    df = df.fillna({
        "user": "unknown",
        "port": 0,
        "bytes": 0,
    })
    df["user"] = df["user"].astype(str)
    df["source_ip"] = df["source_ip"].astype(str)
    df["event_type"] = df["event_type"].astype(str)
    df["port"] = pd.to_numeric(df["port"], errors="coerce").fillna(0).astype(int)
    df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce").fillna(0).astype(int)

    return df


def get_label_distribution(df):
    """Return class distribution for training data."""
    if "label" not in df.columns:
        return {}
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    return {
        label: {
            "count": int(count),
            "percent": round(count / total * 100, 2),
        }
        for label, count in counts.items()
    }