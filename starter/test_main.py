from fastapi.testclient import TestClient
from starter.main import app


client = TestClient(app)


def test_get_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Census Income Prediction API"
    }


def test_post_prediction_low_income():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    response = client.post("/model/", json=data)

    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]


def test_post_prediction_high_income():
    data = {
        "age": 52,
        "workclass": "Private",
        "fnlgt": 209642,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States",
    }

    response = client.post("/model/", json=data)

    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]
