Gojek Driver Allocation – Machine Learning Pipeline
1. Project Overview

This project replicates a real-world machine learning workflow used for driver allocation at Gojek.
The goal is to develop a complete data pipeline that:

Cleans and processes raw booking and driver data

Engineers meaningful features

Trains a classification model to identify the best driver for an order

Generates predictions for new orders

Provides evaluation metrics for the model

The structure and workflow closely follow industry ML practices.

2. Dataset Description

The project uses three main input files:

booking_log.csv – Booking and order-related events

participant_log.csv – Driver-related attributes

test_data.csv – Orders for which predictions must be generated

These files contain features such as event timestamps, location data, driver statistics, and booking information.

3. Project Structure
ds-assignment-master/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── utils/
│
├── test/
│   └── test_features/
│   └── test_utils/
│
├── submission/
│
├── environment.yaml
├── Makefile
└── README.md

4. Pipeline Stages
Step 1: Data Creation

make data
Generates cleaned and merged datasets from raw logs.

Step 2: Feature Engineering

make features
Applies transformations such as distance calculation, time extraction, and derived attributes.

Step 3: Model Training

make train
Trains a classification model (Random Forest or XGBoost depending on configuration).

Step 4: Prediction

make predict
Generates driver allocation predictions for new orders.

Step 5: Testing

pytest
Runs unit tests for feature engineering and utilities.

5. Models Used

The project uses:

Random Forest Classifier

XGBoost Classifier

The final model may vary depending on performance metrics during experimentation.

6. Results Summary

Model performance evaluated using metrics such as Accuracy, ROC-AUC, Precision, Recall, and Log Loss.

Final outputs include:

metrics.json – Model performance scores

results.csv – Predicted driver allocations for test orders

7. Challenges and Learnings

Some key challenges addressed during development:

Handling missing values and inconsistent data formats

Merging booking and participant logs correctly

Engineering meaningful features from timestamps and geo-coordinates

Fixing pipeline failures caused by path issues or incorrect environment setups

Updating deprecated libraries and test frameworks

Key learnings include working with ML pipelines, debugging production-style code, and improving feature engineering design.

8. Future Improvements

Hyperparameter tuning using GridSearch or Optuna

Deployment-ready packaging

More advanced models: LightGBM, CatBoost

Handling real-time streaming data

Improving test coverage and monitoring

9. How to Run the Project
Create Conda Environment
conda create -n work-at-gojek python=3.12 -y
conda activate work-at-gojek

Install Required Libraries
pip install pandas numpy scikit-learn xgboost matplotlib python-dateutil pytz tzdata pytest
conda install -c conda-forge make -y

Run Pipeline
make data
make features
make train
make predict
pytest -q
make run
