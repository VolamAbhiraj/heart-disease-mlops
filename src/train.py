import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------
# FIX FOR CI (GitHub Actions) – force local MLflow tracking directory
# ------------------------------------------------------------------
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

# Load dataset
df = pd.read_csv("data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Convert all columns to numeric
X = X.apply(pd.to_numeric)

# ---- Convert to Binary Target ----
# Any value >0 means presence of disease
y = y.apply(lambda v: 1 if v > 0 else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Heart Disease Prediction")

# ================= LOGISTIC REGRESSION =================
with mlflow.start_run(run_name="Logistic Regression"):

    lr_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    lr_pipeline.fit(X_train, y_train)
    preds = lr_pipeline.predict(X_test)

    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("max_iter", 2000)

    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("precision", precision_score(y_test, preds))
    mlflow.log_metric("recall", recall_score(y_test, preds))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, preds))

    # Safe model logging (works locally + CI)
    mlflow.sklearn.log_model(lr_pipeline, "logistic-model")

    joblib.dump(lr_pipeline, "src/logistic_pipeline.pkl")

# ================= RANDOM FOREST =================
with mlflow.start_run(run_name="Random Forest"):

    rf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    rf_pipeline.fit(X_train, y_train)
    preds = rf_pipeline.predict(X_test)

    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("precision", precision_score(y_test, preds))
    mlflow.log_metric("recall", recall_score(y_test, preds))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, preds))

    mlflow.sklearn.log_model(rf_pipeline, "rf-model")

    joblib.dump(rf_pipeline, "src/rf_pipeline.pkl")
