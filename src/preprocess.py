import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess(path="data/heart.csv"):
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "src/scaler.pkl")
    return X_scaled, df["target"]
