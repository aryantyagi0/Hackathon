📘 Gojek Driver Booking Acceptance Prediction
A Machine Learning Pipeline with Feature Engineering & XGBoost

This project predicts whether a driver will accept or reject a booking request based on booking data and driver historical behavior.
It follows a clean, modular ML pipeline inspired by real-world production systems.

📂 Project Structure
.
├── config.toml
├── data
│   ├── raw
│   │   ├── booking_log.csv
│   │   ├── participant_log.csv
│   └── processed
│       ├── dataset.csv
│       └── transformed_dataset.csv
├── models
│   └── saved_model.pkl
├── submission
│   ├── metrics.json
│   └── results.csv
├── src
│   ├── data
│   │   └── make_dataset.py
│   ├── features
│   │   ├── build_features.py
│   │   └── transformations.py
│   ├── models
│   │   ├── classifier.py
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── utils
│       ├── config.py
│       ├── guardrails.py
│       ├── store.py
│       └── time.py
└── test

🎯 Objective

Predict:

is_completed = 1 if driver accepts booking  
is_completed = 0 otherwise


This helps improve dispatch efficiency in ride-hailing systems.

🧹 1. Data Preparation

File: src/data/make_dataset.py

Steps:

Load raw logs (booking_log.csv, participant_log.csv)

Remove duplicates

Merge booking + participant data

Create binary target column:

is_completed = participant_status == "ACCEPTED"


Output → data/processed/dataset.csv

🧠 2. Feature Engineering

File: src/features/transformations.py

The following features were engineered:

✔ driver_distance

Using haversine distance between driver & pickup location

✔ event_hour

Hour extracted from event timestamp

✔ driver_historical_completed_bookings

How many bookings driver completed before current event
→ Captures reliability pattern

✔ driver_accept_rate

Driver's historical acceptance ratio
→ Very strong predictor of outcome (behavioral signal)

Output → data/processed/transformed_dataset.csv

🤖 3. Model Training

File: src/models/train_model.py

Model Used: XGBoostClassifier

Why XGBoost?

Handles class imbalance using scale_pos_weight

Captures nonlinear relationships

Performs better on tabular data compared to RandomForest

Key Parameters:
n_estimators = 500
max_depth = 6
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
scale_pos_weight = 4
eval_metric = "logloss"


Model wrapper → SklearnClassifier

📊 4. Evaluation Metrics

Implemented in: src/models/classifier.py

Includes:

Accuracy

Precision

Recall

F1 Score

ROC–AUC

Output → submission/metrics.json

Example:

{
  "accuracy": 0.58,
  "precision": 0.55,
  "recall": 0.48,
  "f1_score": 0.49,
  "roc_auc": 0.62
}

📈 5. Predictions

File: src/models/predict_model.py

Generates:

submission/results.csv


Which contains the predicted probability of booking acceptance.

🏃 Running the Pipeline
1️⃣ Activate environment
conda activate gojek

2️⃣ Build dataset
python -m src.data.make_dataset

3️⃣ Build features
python -m src.features.build_features

4️⃣ Train model
python -m src.models.train_model

5️⃣ Generate predictions
python -m src.models.predict_model

⚙️ Configuration (config.toml)
features = [
    "trip_distance",
    "driver_distance",
    "event_hour",
    "driver_gps_accuracy"
]

target = "is_completed"
test_size = 0.2

[random_forest]
class_weight = "balanced"


Note: XGBoost was used directly in train_model.py instead of RF.

🧪 Tests

Found in /test directory
Includes:

Feature transformation tests

Utility function tests

Store I/O tests

🚀 Key Improvements Implemented

Advanced feature engineering

Behavioral driver features (accept rate, history)

XGBoost model replacing RandomForest

Improved evaluation metrics

Full modular ML pipeline

Clean folder structure

Config-driven architecture

🧑‍💻 Author

Aryan Tyagi
Machine Learning & Data Science Engineer
GitHub: aryantyagi0
