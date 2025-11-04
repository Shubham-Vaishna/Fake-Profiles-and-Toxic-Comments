"""
pr.py
Cleaned training script for Fake Profile ANN.
- Expects a sanitized demo CSV at data/sample_fake_profiles.csv
- Outputs model to models/ann_fake_profile.h5 and eval to results/fake_profile_eval.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Paths (relative to repo root)
SAMPLE_CSV = os.path.join("data", "sample_fake_profiles.csv")
MODEL_DIR = "models"
RESULTS_DIR = "results"
MODEL_FILE = os.path.join(MODEL_DIR, "ann_fake_profile.h5")
EVAL_FILE = os.path.join(RESULTS_DIR, "fake_profile_eval.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(csv_path=SAMPLE_CSV):
    """
    Load sample fake profile CSV.
    Expected: features columns (numerical) and a target column named 'is_fake' (0/1).
    If target name differs, edit this loader accordingly.
    """
    df = pd.read_csv(csv_path)
    # Basic sanitization: drop obvious PII columns if present
    for pii in ["user_id", "email", "username", "ip"]:
        if pii in df.columns:
            df = df.drop(columns=pii)
    # Ensure target column exists
    if "is_fake" not in df.columns:
        raise ValueError("Expected target column 'is_fake' in sample CSV.")
    return df

def preprocess(df):
    """
    Simple preprocessing: separate features/target, fillna, scale numeric features.
    """
    X = df.drop(columns=["is_fake"])
    y = df["is_fake"].astype(int).to_numpy()

    # Keep only numeric columns for this demo; encode categoricals if needed
    X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
    # If you have categorical columns, add encoding here (one-hot or ordinal).

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    return X_scaled, y, scaler, X_numeric.columns.tolist()

def build_model(input_dim, hidden_units=[64, 32], dropout=0.2):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for h in hidden_units:
        model.add(layers.Dense(h, activation="relu"))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_and_evaluate(X, y, epochs=30, batch_size=32, validation_split=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_model(X_train.shape[1])
    cb = [callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    history = model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs,
                        batch_size=batch_size, callbacks=cb, verbose=1)
    preds = (model.predict(X_test) > 0.5).astype(int).ravel()
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    # Save model and simple results
    model.save(MODEL_FILE)
    # Save summary
    summary = {
        "accuracy": [acc],
        "n_train": [len(y_train)],
        "n_test": [len(y_test)]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(EVAL_FILE, index=False)

    return model, history, report, cm

def main():
    print("Loading sample fake profile data from:", SAMPLE_CSV)
    df = load_data(SAMPLE_CSV)
    X, y, scaler, feature_names = preprocess(df)
    print(f"Features used: {len(feature_names)} numeric features")
    model, history, report, cm = train_and_evaluate(X, y, epochs=20)

    print("Training complete. Evaluation summary:")
    print("Confusion Matrix:\n", cm)
    print("Classification report (summary):")
    for k, v in report.items():
        if k in ("0", "1", "macro avg", "weighted avg"):
            print(k, v)
    print(f"Saved model to {MODEL_FILE}")
    print(f"Saved eval summary to {EVAL_FILE}")

if __name__ == "__main__":
    main()
