"""
AI-Powered Lead Generation - Lead Scoring
=========================================
End-to-end pipeline that scores leads by their probability of
converting, so a sales team can call the hottest leads first.

Dataset : UCI Bank Marketing dataset (Moro et al., 2014) - records of
          a Portuguese bank's phone-marketing campaigns.
          Here the target `y` ("client subscribed a term deposit",
          yes/no) is used as the *conversion* outcome: each contacted
          client is treated as a lead, and the model learns which
          leads convert. This is public benchmark data, not company data.
Source  : https://archive.ics.uci.edu/dataset/222/bank+marketing
          (file: bank-full.csv inside the dataset zip)

Important modelling note
------------------------
The `duration` column (length of the last call, in seconds) is EXCLUDED.
It is only known *after* the call ends, so a real lead-scoring system
cannot use it to prioritise leads beforehand - the dataset authors
themselves flag it as leakage for realistic prediction.

Pipeline steps
--------------
1. Load and clean the data (';' separated CSV, target y -> 0/1).
2. Quick exploratory summary (shape, conversion rate).
3. Preprocessing: one-hot encoding for categoricals, scaling for numerics.
4. Stratified train/test split (80/20).
5. Train Logistic Regression, Random Forest and XGBoost.
6. Evaluate (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).
7. Lead scoring: convert the best model's probabilities into a
   0-100 score with Hot / Warm / Cold bands, score the test leads,
   and save the top-scoring leads to results/sample_scored_leads.csv.
8. Save all metrics to results/metrics.json and the best model to disk.

Run:  python lead_scoring.py
"""

import io
import json
import os
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # XGBoost is optional
    HAS_XGB = False

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "bank-full.csv")
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RESULTS_PATH = os.path.join("results", "metrics.json")
SAMPLE_PATH = os.path.join("results", "sample_scored_leads.csv")
MODEL_PATH = os.path.join("models", "best_lead_scoring_model.pkl")

# Band thresholds for the 0-100 lead score.
HOT_THRESHOLD = 70
WARM_THRESHOLD = 40


# ----------------------------------------------------------------------
# 1. Load & clean
# ----------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load bank-full.csv (download + unzip it first if it is missing).

    The UCI zip nests a second zip (bank.zip) which contains
    bank-full.csv; both layers are handled here.
    """
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found locally - downloading from:\n  {DATA_URL}")
        os.makedirs("data", exist_ok=True)
        raw = urllib.request.urlopen(DATA_URL).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as outer:
            inner_bytes = outer.read("bank.zip")
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
            csv_bytes = inner.read("bank-full.csv")
        with open(DATA_PATH, "wb") as f:
            f.write(csv_bytes)

    df = pd.read_csv(DATA_PATH, sep=";")
    # Binary target: subscribed a term deposit -> 1 (converted), else 0.
    df["converted"] = (df["y"] == "yes").astype(int)
    df = df.drop(columns=["y"])
    # 'duration' is known only after the call -> leakage, drop it.
    df = df.drop(columns=["duration"])
    return df


# ----------------------------------------------------------------------
# 2. EDA summary
# ----------------------------------------------------------------------
def eda_summary(df: pd.DataFrame) -> dict:
    conv_rate = df["converted"].mean()
    print("\n--- EDA summary ---")
    print(f"Shape                 : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Converted leads (1)   : {int(df['converted'].sum())}")
    print(f"Not converted (0)     : {int((df['converted'] == 0).sum())}")
    print(f"Conversion rate       : {conv_rate:.2%}")
    print(f"Age                   : min {df['age'].min()}, "
          f"median {df['age'].median()}, max {df['age'].max()}")
    print(f"Avg balance (EUR)     : converters "
          f"{df.loc[df.converted == 1, 'balance'].mean():.0f} vs "
          f"non-converters {df.loc[df.converted == 0, 'balance'].mean():.0f}")
    print(f"Missing values total  : {int(df.isna().sum().sum())}")
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "conversion_rate": round(float(conv_rate), 4),
        "converted_leads": int(df["converted"].sum()),
    }


# ----------------------------------------------------------------------
# Lead scoring helpers
# ----------------------------------------------------------------------
def score_to_band(score: float) -> str:
    """Map a 0-100 lead score onto a Hot / Warm / Cold band."""
    if score >= HOT_THRESHOLD:
        return "Hot"
    if score >= WARM_THRESHOLD:
        return "Warm"
    return "Cold"


def score_leads(model, X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X with conversion probability, 0-100 lead
    score and Hot/Warm/Cold band columns added."""
    proba = model.predict_proba(X)[:, 1]
    scored = X.copy()
    scored["conversion_probability"] = np.round(proba, 4)
    scored["lead_score"] = np.round(proba * 100).astype(int)
    scored["band"] = [score_to_band(s) for s in scored["lead_score"]]
    return scored


# ----------------------------------------------------------------------
# 3-6. Preprocessing, training, evaluation
# ----------------------------------------------------------------------
def build_pipeline(model, numeric_features, categorical_features) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main() -> None:
    df = load_data()
    dataset_info = eda_summary(df)

    X = df.drop(columns=["converted"])
    y = df["converted"]
    numeric_features = ["age", "balance", "day", "campaign", "pdays", "previous"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # Only ~11.7% of leads convert, so class weighting matters here too.
    neg, pos = np.bincount(y_train)
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=5,
            scale_pos_weight=neg / pos,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    results = {}
    trained = {}
    print("\n--- Model results (test set) ---")
    for name, model in models.items():
        pipe = build_pipeline(model, numeric_features, categorical_features)
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        trained[name] = pipe
        print(
            f"{name:20s} acc={metrics['accuracy']:.4f}  "
            f"prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}  "
            f"f1={metrics['f1_score']:.4f}  roc_auc={metrics['roc_auc']:.4f}"
        )

    best_name = max(results, key=lambda n: results[n]["f1_score"])
    best_model = trained[best_name]
    print(f"\nBest model by F1-score: {best_name} "
          f"(F1 = {results[best_name]['f1_score']:.4f})")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: "
          f"{results[best_name]['confusion_matrix']}")

    # ------------------------------------------------------------------
    # 7. Score the held-out leads with the best model
    # ------------------------------------------------------------------
    scored = score_leads(best_model, X_test)
    scored["actual_converted"] = y_test.values
    band_counts = scored["band"].value_counts().to_dict()
    print("\n--- Lead-score bands on the test set ---")
    for band in ["Hot", "Warm", "Cold"]:
        subset = scored[scored["band"] == band]
        share_converted = subset["actual_converted"].mean() if len(subset) else 0.0
        print(f"{band:5s}: {len(subset):5d} leads, "
              f"actual conversion rate in band: {share_converted:.2%}")

    top_leads = scored.sort_values("lead_score", ascending=False).head(10)
    display_cols = [
        "age", "job", "marital", "education", "balance", "housing",
        "loan", "contact", "poutcome", "conversion_probability",
        "lead_score", "band", "actual_converted",
    ]
    print("\nTop 10 scored leads (test set):")
    print(top_leads[display_cols].to_string(index=False))

    # ------------------------------------------------------------------
    # 8. Persist results + model artifact
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    top_leads[display_cols].to_csv(SAMPLE_PATH, index=False)
    report = {
        "project": "AI-Powered Lead Generation (Lead Scoring)",
        "dataset": {
            "name": "UCI Bank Marketing (bank-full.csv)",
            "source_url": DATA_URL,
            "target_definition": (
                "y = client subscribed a term deposit (yes=1) after a "
                "marketing contact; used as the lead-conversion outcome"
            ),
            "excluded_features": {
                "duration": (
                    "last call duration in seconds - only known after the "
                    "call ends, so it is leakage for lead prioritisation"
                )
            },
            **dataset_info,
        },
        "split": {
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "test_fraction": 0.20,
            "random_state": RANDOM_STATE,
        },
        "models": results,
        "best_model": best_name,
        "best_model_metrics": results[best_name],
        "lead_score_bands": {
            "Hot": f"score >= {HOT_THRESHOLD}",
            "Warm": f"{WARM_THRESHOLD} <= score < {HOT_THRESHOLD}",
            "Cold": f"score < {WARM_THRESHOLD}",
            "test_set_band_counts": {k: int(v) for k, v in band_counts.items()},
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved metrics -> {RESULTS_PATH}")
    print(f"Saved top scored leads -> {SAMPLE_PATH}")
    print(f"Saved best model ({best_name}) -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
