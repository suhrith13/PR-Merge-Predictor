"""
PR Merge Predictor — Model Training Script

Trains a logistic regression model on the labeled dataset produced by
collect_data.py, evaluates it honestly against the naive "always predict
the repo's baseline merge rate" baseline, and exports the trained
coefficients as plain JSON so they can be embedded directly in a
Cloudflare Worker with zero ML runtime dependency.

Usage:
    python train_model.py --data ../data/prs.csv --out ../model/weights.json
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss

NUMERIC_FEATURES = [
    "additions", "deletions", "total_diff", "changed_files",
    "title_length", "body_length", "has_test_keyword",
    "is_first_time_contributor", "contributor_merge_rate",
    "day_of_week", "hour_of_day",
]

# author_association is categorical -> one-hot encode
CATEGORICAL_FEATURES = ["author_association"]


def load_and_prepare(path: str):
    df = pd.read_csv(path)
    df = df.dropna(subset=["label_merged"])

    # Clip extreme outliers (a PR with a 100k-line diff will otherwise
    # dominate the scaler) -- cap at the 99th percentile
    for col in ["additions", "deletions", "total_diff", "changed_files"]:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix="assoc")
    dummy_cols = [c for c in df.columns if c.startswith("assoc_")]

    feature_cols = NUMERIC_FEATURES + dummy_cols
    X = df[feature_cols].fillna(0)
    y = df["label_merged"]

    return X, y, feature_cols, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    X, y, feature_cols, df = load_and_prepare(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    # --- Honest evaluation ---
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_pred_proba)

    # Naive baseline: always predict the majority class
    majority_class = y_train.mode()[0]
    baseline_pred = np.full_like(y_test, majority_class)
    baseline_acc = (baseline_pred == y_test).mean()
    model_acc = (y_pred == y_test).mean()

    print("=" * 60)
    print("EVALUATION (held-out test set, never seen during training)")
    print("=" * 60)
    print(f"  Rows total       : {len(df)}")
    print(f"  Train / test     : {len(X_train)} / {len(X_test)}")
    print(f"  Naive baseline acc (always predict '{majority_class}') : {baseline_acc:.3f}")
    print(f"  Model accuracy    : {model_acc:.3f}")
    print(f"  Model AUC         : {auc:.3f}   (0.5 = random, 1.0 = perfect)")
    print(f"  Model precision   : {precision:.3f}")
    print(f"  Model recall      : {recall:.3f}")
    print(f"  Brier score       : {brier:.3f}   (lower = better calibrated, 0 = perfect)")
    print("=" * 60)

    if auc < 0.6:
        print("WARNING: AUC is close to random. Collect more data or revisit features before shipping.")

    # --- Export weights for the Worker (pure JS dot-product, no ML runtime needed) ---
    export = {
        "feature_order": feature_cols,
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "eval_metrics": {
            "auc": round(auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "brier_score": round(brier, 4),
            "baseline_accuracy": round(baseline_acc, 4),
            "model_accuracy": round(model_acc, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
    }

    with open(args.out, "w") as f:
        json.dump(export, f, indent=2)

    print(f"\nExported model weights to {args.out}")
    print("Copy this file's contents into the Cloudflare Worker to serve live predictions.")


if __name__ == "__main__":
    main()
