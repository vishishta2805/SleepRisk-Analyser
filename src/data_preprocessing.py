import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    X = df.drop("Sleep Disorder", axis=1)
    y = df["Sleep Disorder"]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)