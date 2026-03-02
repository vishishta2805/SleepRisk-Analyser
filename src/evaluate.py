import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix


def evaluate():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    MODEL_PATH = os.path.join(BASE_DIR, "models", "pipeline.pkl")
    TEST_X_PATH = os.path.join(BASE_DIR, "data", "processed", "X_test.csv")
    TEST_Y_PATH = os.path.join(BASE_DIR, "data", "processed", "y_test.csv")

    # Load model
    pipeline = joblib.load(MODEL_PATH)

    # Load saved test set
    X_test = pd.read_csv(TEST_X_PATH)
    y_test = pd.read_csv(TEST_Y_PATH).squeeze()

    # Predict
    predictions = pipeline.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    evaluate()