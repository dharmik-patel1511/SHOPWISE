"""
SHOPWISE — E-Commerce Conversion Intelligence
IBM Internship Project

Dataset: UCI Online Shoppers Purchasing Intention Dataset (ID 468)
Official source:
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    dataset = fetch_ucirepo(id=468)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    df = X.copy()
    df["Revenue"] = y
    return df


def clean_and_engineer(df):
    df = df.copy()

    for col in ["Weekend", "Revenue"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.upper()
                .map({"TRUE": 1, "FALSE": 0})
            )

    page_cols = ["Administrative", "Informational", "ProductRelated"]
    duration_cols = [
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
    ]

    df["TotalPages"] = df[page_cols].sum(axis=1)
    df["TotalDuration"] = df[duration_cols].sum(axis=1)
    df["AvgSecondsPerPage"] = np.where(
        df["TotalPages"] > 0,
        df["TotalDuration"] / df["TotalPages"],
        0,
    )
    df["EngagementScore"] = (
        np.log1p(df["TotalPages"])
        + np.log1p(df["TotalDuration"]) / 5
        + df["PageValues"]
    )
    return df


def save_business_outputs(df):
    sessions = len(df)
    purchasers = int(df["Revenue"].sum())
    conversion_rate = purchasers / sessions if sessions else 0

    kpis = pd.DataFrame({
        "KPI": ["Sessions", "Purchasers", "Conversion Rate", "Avg Page Value"],
        "Value": [
            sessions,
            purchasers,
            round(conversion_rate * 100, 2),
            round(df["PageValues"].mean(), 4),
        ],
    })
    kpis.to_csv(OUTPUT_DIR / "kpis.csv", index=False)

    visitor = (
        df.groupby("VisitorType")["Revenue"]
        .agg(["count", "sum", "mean"])
        .rename(columns={
            "count": "sessions",
            "sum": "purchasers",
            "mean": "conversion_rate",
        })
        .reset_index()
    )
    visitor["conversion_rate"] *= 100
    visitor.to_csv(OUTPUT_DIR / "visitor_type_performance.csv", index=False)

    monthly = (
        df.groupby("Month")["Revenue"]
        .agg(["count", "sum", "mean"])
        .rename(columns={
            "count": "sessions",
            "sum": "purchasers",
            "mean": "conversion_rate",
        })
        .reset_index()
    )
    monthly["conversion_rate"] *= 100
    monthly.to_csv(OUTPUT_DIR / "monthly_performance.csv", index=False)

    traffic = (
        df.groupby("TrafficType")["Revenue"]
        .agg(["count", "sum", "mean"])
        .rename(columns={
            "count": "sessions",
            "sum": "purchasers",
            "mean": "conversion_rate",
        })
        .reset_index()
        .sort_values("conversion_rate", ascending=False)
    )
    traffic["conversion_rate"] *= 100
    traffic.to_csv(OUTPUT_DIR / "traffic_performance.csv", index=False)

    plt.figure(figsize=(7, 4))
    (
        df["Revenue"].value_counts()
        .sort_index()
        .rename({0: "No Purchase", 1: "Purchase"})
        .plot(kind="bar")
    )
    plt.title("Purchase Outcome Distribution")
    plt.ylabel("Sessions")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "purchase_outcome.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    visitor.set_index("VisitorType")["conversion_rate"].sort_values().plot(
        kind="barh"
    )
    plt.title("Conversion Rate by Visitor Type")
    plt.xlabel("Conversion Rate (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "visitor_conversion.png", dpi=160)
    plt.close()

    month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_plot = monthly.copy()
    month_plot["Month"] = pd.Categorical(
        month_plot["Month"], categories=month_order, ordered=True
    )
    month_plot = month_plot.sort_values("Month")

    plt.figure(figsize=(9, 4))
    plt.plot(
        month_plot["Month"].astype(str),
        month_plot["conversion_rate"],
        marker="o",
    )
    plt.title("Monthly Conversion Rate")
    plt.ylabel("Conversion Rate (%)")
    plt.xlabel("Month")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_conversion.png", dpi=160)
    plt.close()

    return kpis, visitor, monthly, traffic


def train_models(df):
    X = df.drop(columns=["Revenue"])
    y = df["Revenue"].astype(int)

    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    rows = []
    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]

        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        })

    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    return metrics


def main():
    print("Loading UCI Online Shoppers Purchasing Intention Dataset...")
    df = load_data()
    print(f"Raw shape: {df.shape}")

    df = clean_and_engineer(df)
    kpis, visitor, monthly, traffic = save_business_outputs(df)
    metrics = train_models(df)

    print("\n=== BUSINESS KPIs ===")
    print(kpis.to_string(index=False))

    print("\n=== VISITOR TYPE PERFORMANCE ===")
    print(visitor.to_string(index=False))

    print("\n=== TOP TRAFFIC TYPES BY CONVERSION ===")
    print(traffic.head(10).to_string(index=False))

    print("\n=== MODEL METRICS ===")
    print(metrics.to_string(index=False))

    print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
