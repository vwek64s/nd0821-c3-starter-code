import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data
from starter.starter.ml.model import compute_model_metrics, inference, train_model


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


def clean_data(data):
    data.columns = data.columns.str.strip()

    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].str.strip()

    return data


def compute_slice_metrics(test, feature, model, encoder, lb):
    lines = []

    for value in sorted(test[feature].unique()):
        slice_data = test[test[feature] == value]

        X_slice, y_slice, _, _ = process_data(
            slice_data,
            categorical_features=CAT_FEATURES,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)

        lines.extend(
            [
                f"{feature}: {value}",
                f"precision: {precision}",
                f"recall: {recall}",
                f"fbeta: {fbeta}",
                "",
            ]
        )

    return lines


def write_slice_output(test, model, encoder, lb, output_path):
    lines = []

    for feature in CAT_FEATURES:
        lines.append(f"Feature: {feature}")
        lines.append("=" * 80)
        lines.extend(compute_slice_metrics(test, feature, model, encoder, lb))
        lines.append("")

    with open(output_path, "w") as file:
        file.write("\n".join(lines))


def main():
    data_path = os.path.join("starter", "data", "census.csv")
    model_dir = "model"
    slice_output_path = "slice_output.txt"

    os.makedirs(model_dir, exist_ok=True)

    data = pd.read_csv(data_path)
    data = clean_data(data)

    train, test = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data["salary"],
    )

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Fbeta: {fbeta}")

    write_slice_output(
        test=test,
        model=model,
        encoder=encoder,
        lb=lb,
        output_path=slice_output_path,
    )

    with open(os.path.join(model_dir, "model.pkl"), "wb") as file:
        pickle.dump(model, file)

    with open(os.path.join(model_dir, "encoder.pkl"), "wb") as file:
        pickle.dump(encoder, file)

    with open(os.path.join(model_dir, "lb.pkl"), "wb") as file:
        pickle.dump(lb, file)


if __name__ == "__main__":
    main()
