# src/models/classifier.py

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class Classifier(ABC):
    """Abstract base classifier"""

    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnClassifier(Classifier):
    """Wrapper around any sklearn classifier"""

    def __init__(self, estimator: BaseEstimator, features: List[str], target: str):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(
            df_train[self.features].values,
            df_train[self.target].values,
        )

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        y_true = df_test[self.target].values
        y_pred_proba = self.clf.predict_proba(df_test[self.features].values)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        return {
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(df[self.features].values)[:, 1]
