import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report


def train():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_PATH = os.path.join(
        BASE_DIR,
        "data",
        "raw",
        "Sleep_health_and_lifestyle_dataset.csv"
    )

    df = pd.read_csv(DATA_PATH)

    # ---------------- CLEANING ----------------
    df = df.drop_duplicates()
    df = df.dropna(subset=["Sleep Disorder"])

    df = df.drop(columns=["Cluster", "Person ID"], errors="ignore")

    if "Blood Pressure" in df.columns:
        df[["Systolic", "Diastolic"]] = (
            df["Blood Pressure"]
            .str.split("/", expand=True)
            .astype(float)
        )
        df = df.drop(columns=["Blood Pressure"])

    if "BMI Category" in df.columns:
        df["BMI Category"] = df["BMI Category"].replace(
            {"Normal Weight": "Normal"}
        )

    if "Physical Activity Level" in df.columns and "Daily Steps" in df.columns:
        df["Activity_Index"] = (
            df["Physical Activity Level"] * df["Daily Steps"]
        )
        df = df.drop(columns=["Physical Activity Level", "Daily Steps"])

    # ---------------- FEATURE / TARGET ----------------
    X = df.drop("Sleep Disorder", axis=1)
    y = df["Sleep Disorder"]

    # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Save test set
    processed_path = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(processed_path, exist_ok=True)

    X_test.to_csv(os.path.join(processed_path, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(processed_path, "y_test.csv"), index=False)

    # ---------------- PREPROCESSING ----------------
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])

    # ---------------- TRAIN ----------------
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print("\nModel Evaluation:\n")
    print(classification_report(y_test, predictions))

    # ---------------- SAVE MODEL ----------------
    MODEL_PATH = os.path.join(BASE_DIR, "models", "pipeline.pkl")
    joblib.dump(pipeline, MODEL_PATH)

    print("\nModel saved successfully!")
    print("Test set saved in data/processed/")


if __name__ == "__main__":
    train()