import os
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference


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

MODEL_DIR = "model"


def load_pickle(filename):
    with open(os.path.join(MODEL_DIR, filename), "rb") as file:
        return pickle.load(file)


model = load_pickle("model.pkl")
encoder = load_pickle("encoder.pkl")
lb = load_pickle("lb.pkl")

app = FastAPI()


class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
        },
    )


@app.get("/")
def read_root():
    return {"message": "Welcome to the Census Income Prediction API"}


@app.post("/model/")
def predict(data: CensusData):
    input_df = pd.DataFrame(
        [
            data.model_dump(by_alias=True)
        ]
    )

    X, _, _, _ = process_data(
        input_df,
        categorical_features=CAT_FEATURES,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    prediction = inference(model, X)
    prediction_label = lb.inverse_transform(prediction)[0]

    return {"prediction": prediction_label}
