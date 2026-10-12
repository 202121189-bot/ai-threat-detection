#!/usr/bin/env python3
"""
Threat Predictor
=================
Loads the trained model and predicts whether new events are
malicious or benign.

Pipeline:
  1. Load new events (CSV)
  2. Load saved model + preprocessor
  3. Predict labels + probabilities
  4. Generate HTML/JSON/CSV report

Outputs:
  - reports/prediction_YYYYMMDD_HHMMSS/report.html
  - reports/prediction_YYYYMMDD_HHMMSS/predictions.csv
  - reports/prediction_YYYYMMDD_HHMMSS/summary.json
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from jinja2 import Environment, FileSystemLoader

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_new_events
from src.features import build_feature_frame, load_pipeline


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


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------

def predict(events_file, model_dir="models", output_dir=None):
    banner("[1/5] Loading model artifacts")
    clf_path = os.path.join(model_dir, "threat_classifier.pkl")
    prep_path = os.path.join(model_dir, "preprocessor.pkl")

    if not os.path.exists(clf_path):
        print(f"[!] Trained model not found: {clf_path}")
        print("    Run `python src/train.py` first.")
        sys.exit(1)

    clf = joblib.load(clf_path)
    preprocessor = load_pipeline(prep_path)
    print(f"  [+] Loaded classifier:   {clf_path}")
    print(f"  [+] Loaded preprocessor: {prep_path}")

    # ------------------------------------------------
    banner("[2/5] Loading new events")
    df = load_new_events(events_file)
    print(f"  [+] Events to score: {len(df)}")

    # ------------------------------------------------
    banner("[3/5] Building features & predicting")
    X_raw = build_feature_frame(df)
    X_proc = preprocessor.transform(X_raw)
    preds = clf.predict(X_proc)
    probs = clf.predict_proba(X_proc)

    # Attach predictions
    df = df.copy()
    df["prediction"] = ["malicious" if p == 1 else "benign" for p in preds]
    df["confidence"] = [round(float(max(pr)), 4) for pr in probs]

    n_malicious = int((preds == 1).sum())
    n_benign = int((preds == 0).sum())
    rate = round(n_malicious / len(df) * 100, 2) if len(df) else 0.0

    print(f"  [+] Malicious: {n_malicious}")
    print(f"  [+] Benign:    {n_benign}")
    print(f"  [+] Rate:      {rate}%")

    # ------------------------------------------------
    banner("[4/5] Preparing output")
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"reports/prediction_{ts}"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Predictions CSV
    prediction_rows = []
    for _, row in df.iterrows():
        prediction_rows.append({
            "timestamp": row.get("timestamp", ""),
            "event_type": row.get("event_type", ""),
            "source_ip": row.get("source_ip", ""),
            "user": row.get("user", ""),
            "port": row.get("port", ""),
            "bytes": row.get("bytes", ""),
            "prediction": row["prediction"],
            "confidence": row["confidence"],
        })
    write_csv(out / "predictions.csv", prediction_rows)

    # Summary
    summary = {
        "scored_at": datetime.now().isoformat(timespec="seconds"),
        "events_file": events_file,
        "model_dir": model_dir,
        "total_events": len(df),
        "malicious": n_malicious,
        "benign": n_benign,
        "malicious_rate_percent": rate,
    }
    write_json(out / "summary.json", summary)

    # ------------------------------------------------
    banner("[5/5] Generating HTML report")
    env = Environment(loader=FileSystemLoader("."))
    try:
        template = env.get_template("reports/template.html")
    except Exception:
        print("  [!] Template not found, skipping HTML.")
        return out, summary

    suspicious = df[df["prediction"] == "malicious"].sort_values(
        "confidence", ascending=False
    )

    html = template.render(
        summary=summary,
        predictions=prediction_rows,
        suspicious=suspicious.to_dict(orient="records"),
        generated_at=summary["scored_at"],
    )

    with open(out / "report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  [+] HTML:  {out}/report.html")
    print(f"  [+] CSV:   {out}/predictions.csv")
    print(f"  [+] JSON:  {out}/summary.json")

    banner("DONE")
    print(f"  Open the report: {out}\\report.html\n")

    return out, summary


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict threats in new log events"
    )
    parser.add_argument(
        "events_file",
        nargs="?",
        default="data/new_events.csv",
        help="Path to events CSV (default: data/new_events.csv)",
    )
    parser.add_argument(
        "-m", "--model-dir",
        default="models",
        help="Directory with trained model (default: models)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for the report",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  AI THREAT DETECTION — PREDICTOR")
    print("=" * 60)
    print(f"  Events file: {args.events_file}")
    print(f"  Model dir:   {args.model_dir}")
    print(f"  Started:     {datetime.now().isoformat(timespec='seconds')}")

    predict(args.events_file, args.model_dir, args.output)


if __name__ == "__main__":
    main()