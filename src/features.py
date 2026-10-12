"""
Feature Engineering
====================
Transforms raw log events (text + numeric) into a numeric feature
matrix suitable for machine learning.

Strategy:
  - One-hot encode 'event_type' (categorical)
  - Hash 'user' and 'source_ip' into binary features
  - Numeric features: port (normalized), bytes (log-scaled)
  - Flag features: is_sensitive_port, is_external_ip, is_root_user

Uses:
  - scikit-learn's ColumnTransformer for reproducibility
  - Save/load the fitted transformer with joblib
"""

import re
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# Sensitive ports commonly targeted by attackers
SENSITIVE_PORTS = {22, 23, 135, 139, 445, 1433, 3389, 5900, 6379, 27017}

# Private IP ranges (RFC1918 + loopback)
PRIVATE_PREFIXES = (
    "10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
    "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
    "127.",
)

# Users that commonly indicate a threat when seen in logs
SENSITIVE_USERS = {"root", "admin", "administrator", "guest"}


def _is_private(ip: str) -> int:
    return int(any(ip.startswith(p) for p in PRIVATE_PREFIXES))


def _is_sensitive_port(port: int) -> int:
    return int(port in SENSITIVE_PORTS)


def _is_sensitive_user(user: str) -> int:
    return int(user.lower() in SENSITIVE_USERS)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw log DataFrame into a feature DataFrame.
    Keeps only numeric + categorical columns used by the pipeline.
    """
    features = pd.DataFrame(index=df.index)

    # Categorical
    features["event_type"] = df["event_type"].astype(str).str.lower()

    # Numeric
    features["port"] = df["port"].astype(int)
    features["log_bytes"] = np.log1p(df["bytes"].astype(float))

    # Binary flags
    features["is_sensitive_port"] = df["port"].apply(_is_sensitive_port)
    features["is_external_ip"] = df["source_ip"].apply(
        lambda ip: 1 - _is_private(str(ip))
    )
    features["is_sensitive_user"] = df["user"].apply(_is_sensitive_user)

    # IP "class" hint (first two octets) as categorical feature
    features["ip_class"] = df["source_ip"].astype(str).apply(
        lambda ip: ".".join(ip.split(".")[:2]) if "." in ip else "unknown"
    )

    return features


def build_pipeline() -> ColumnTransformer:
    """
    Build the full preprocessing pipeline.
      - OneHotEncoder for categorical columns
      - StandardScaler for numeric columns
      - passthrough for binary flags
    """
    categorical_cols = ["event_type", "ip_class"]
    numeric_cols = ["port", "log_bytes"]
    passthrough_cols = [
        "is_sensitive_port", "is_external_ip", "is_sensitive_user",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
            ("flag", "passthrough", passthrough_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def save_pipeline(pipeline, path: str):
    """Persist the fitted pipeline using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: str):
    """Load a fitted pipeline."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline not found: {path}")
    return joblib.load(path)