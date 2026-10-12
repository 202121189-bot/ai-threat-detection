#!/usr/bin/env python3
"""
Model Trainer
==============
Trains a Random Forest classifier to detect malicious log events.

Pipeline:
  1. Load training data (CSV)
  2. Build features (from features.py)
  3. Train Random Forest with cross-validation
  4. Evaluate: precision, recall, F1, confusion matrix
  5. Save the trained model + preprocessor (joblib)
  6. Print a detailed report

Outputs:
  - models/threat_classifier.pkl     ← trained classifier
  - models/preprocessor.pkl          ← fitted preprocessing pipeline
  - models/metrics.json              ← evaluation metrics
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Add project root to path (for `python src/train.py` execution)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_training_data, get_label_distribution
from src.features import build_feature_frame, build_pipeline, save_pipeline


# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------

def banner(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def save_classifier(clf, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(clf, path)


# --------------------------------------------------------------------
# Main training flow
# --------------------------------------------------------------------

def train(filepath, output_dir="models", test_size=0.2, random_state=42):
    banner("[1/6] Loading training data")
    df = load_training_data(filepath)
    print(f"  [+] Rows:  {len(df)}")
    print(f"  [+] Cols:  {list(df.columns)}")

    dist = get_label_distribution(df)
    for label, info in dist.items():
        print(f"      - {label:<12} {info['count']:>4}  ({info['percent']}%)")

    if len(dist) < 2:
        print("[!] Need at least 2 classes to train.")
        sys.exit(1)

    # ------------------------------------------------
    banner("[2/6] Building features")
    X_raw = build_feature_frame(df)
    y = df["label"].str.lower().map({"malicious": 1, "benign": 0}).astype(int)
    print(f"  [+] Feature columns: {list(X_raw.columns)}")
    print(f"  [+] Malicious: {(y == 1).sum()}  |  Benign: {(y == 0).sum()}")

    # ------------------------------------------------
    banner("[3/6] Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  [+] Train: {len(X_train)}   Test: {len(X_test)}")

    # ------------------------------------------------
    banner("[4/6] Training Random Forest")
    preprocessor = build_pipeline()

    # Fit preprocessor on training data
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train_proc, y_train)
    print("  [+] Model trained ✓")

    # Cross-validation
    cv_scores = cross_val_score(clf, X_train_proc, y_train, cv=5, scoring="f1")
    print(f"  [+] 5-fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # ------------------------------------------------
    banner("[5/6] Evaluating on test set")
    y_pred = clf.predict(X_test_proc)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"  [+] Accuracy:  {acc:.4f}")
    print(f"  [+] Precision: {prec:.4f}")
    print(f"  [+] Recall:    {rec:.4f}")
    print(f"  [+] F1 Score:  {f1:.4f}")

    print("\n  Classification Report:")
    report_str = classification_report(
        y_test, y_pred, target_names=["benign", "malicious"], zero_division=0
    )
    for line in report_str.splitlines():
        print(f"    {line}")

    cm = confusion_matrix(y_test, y_pred)
    print("\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    # Feature importance
    try:
        feature_names = preprocessor.get_feature_names_out()
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[-10:][::-1]
        print("\n  Top 10 Feature Importances:")
        for idx in top_idx:
            print(f"    {feature_names[idx]:<45} {importances[idx]:.4f}")
    except Exception as e:
        print(f"  [!] Could not compute feature importance: {e}")

    # ------------------------------------------------
    banner("[6/6] Saving artifacts")
    clf_path = os.path.join(output_dir, "threat_classifier.pkl")
    prep_path = os.path.join(output_dir, "preprocessor.pkl")
    metrics_path = os.path.join(output_dir, "metrics.json")

    save_classifier(clf, clf_path)
    save_pipeline(preprocessor, prep_path)

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "training_file": filepath,
        "total_samples": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "class_distribution": dist,
        "metrics": {
            "accuracy":  round(acc, 4),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1_score":  round(f1, 4),
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std":  round(float(cv_scores.std()), 4),
        },
        "confusion_matrix": {
            "true_negative":  int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive":  int(cm[1][1]),
        },
    }
    write_json(metrics_path, metrics)

    print(f"  [+] Classifier:   {clf_path}")
    print(f"  [+] Preprocessor: {prep_path}")
    print(f"  [+] Metrics:      {metrics_path}")

    banner("DONE")
    print(f"  Model trained successfully at {datetime.now().isoformat(timespec='seconds')}\n")

    return metrics


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train AI Threat Detection model"
    )
    parser.add_argument(
        "training_file",
        nargs="?",
        default="data/threats_training.csv",
        help="Path to training CSV (default: data/threats_training.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        default="models",
        help="Output directory for model artifacts (default: models)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  AI THREAT DETECTION — MODEL TRAINER")
    print("=" * 60)
    print(f"  Training file: {args.training_file}")
    print(f"  Output dir:    {args.output}")
    print(f"  Started:       {datetime.now().isoformat(timespec='seconds')}")

    train(args.training_file, args.output)


if __name__ == "__main__":
    main()