

# add necessary import
import numpy as np
import pandas as pd

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
)

# sample categorical features
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def load_test_data():
    "Load a small portion of census data for testing"
    data = pd.read_csv("data/census.csv")
    return data.sample(200, random_state=42)


# implement the first test. 
def test_train_model_returns_model():
    """Test that train_model return a trained model object"""
    data = load_test_data()
    X, y, encoder, lb = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    model = train_model(X, y)
    assert model is not None
    assert hasattr(model, "predict")


# implement the second test.
def test_inference_output_shape():
    """Test that inference returns predictions of correct shape"""

    data = load_test_data()
    X, y, encoder, lb = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y)

# implement the third test
def test_compute_model_metrics_range():
    """Test that model metrics are within valid range (0-1)"""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
