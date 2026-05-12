# Model Card

## Model Details

This project uses a RandomForestClassifier to predict whether an individual's income is greater than 50K or less than or equal to 50K per year. The model is trained on the cleaned Census Income dataset using a train-test split with 80% of the data for training and 20% for evaluation.

The model uses one-hot encoding for categorical features and a label binarizer for the target label. The trained model, encoder, and label binarizer are saved as pickle files.

## Intended Use

The model is intended for educational purposes as part of a machine learning deployment project. It demonstrates how to train a classification model, evaluate it, expose it through a FastAPI REST API, and deploy it with CI/CD.

The model should not be used for real-world hiring, lending, salary, or eligibility decisions.

## Training Data

The training data comes from the Census Income dataset provided with the project. The dataset contains demographic and employment-related attributes such as age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country.

The data was cleaned by stripping whitespace from column names and string values. The dataset was split into training and test sets using an 80/20 split.

## Evaluation Data

The evaluation data is the 20% holdout test set created from the cleaned Census Income dataset. The split uses a fixed random state and stratification on the salary label to preserve the target distribution across train and test sets.

## Metrics

The model was evaluated using precision, recall, and F-beta score with beta equal to 1.

The model achieved the following results on the holdout test set:

- Precision: 0.7327
- Recall: 0.6397
- F-beta: 0.6830

Additional slice-based metrics were computed for categorical features and saved in `slice_output.txt`.

## Ethical Considerations

The dataset contains sensitive demographic attributes such as sex and race. These features can introduce or amplify bias if the model is used for real-world decision-making. The model may reflect historical and societal inequalities present in the data.

Because of these risks, this model should only be used for learning and demonstration purposes. It should not be used to make decisions that affect people’s employment, compensation, credit, access to services, or legal status.

## Caveats and Recommendations

The model is trained on a limited historical dataset and may not generalize to current populations or real-world production data. The model performance varies across data slices, so aggregate metrics alone are not sufficient to evaluate fairness or reliability.

Before any real-world use, the model would require stronger validation, fairness analysis, bias mitigation, monitoring, explainability, and review by domain experts.
