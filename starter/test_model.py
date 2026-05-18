import numpy as np
from sklearn.ensemble import RandomForestClassifier

from starter.starter.ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)


def test_train_model_returns_model():
    X_train = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ])

    y_train = np.array([0, 1, 0, 1])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_predictions():
    X_train = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ])

    y_train = np.array([0, 1, 0, 1])

    model = train_model(X_train, y_train)

    preds = inference(model, X_train)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y_train)


def test_compute_model_metrics_returns_floats():
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
