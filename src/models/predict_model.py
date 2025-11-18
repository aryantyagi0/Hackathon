import numpy as np
import pandas as pd

from src.features.transformations import driver_distance_to_pickup, hour_of_day
from src.utils.guardrails import validate_prediction_results
from src.utils.store import AssignmentStore


def apply_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply only features allowed during prediction."""
    df = driver_distance_to_pickup(df)
    df = hour_of_day(df)
    return df


@validate_prediction_results
def main():
    store = AssignmentStore()

    print("📥 Loading test data...")
    df_test = store.get_raw("test_data.csv")

    print("⚙️ Applying prediction-time feature engineering...")
    df_test = apply_prediction_features(df_test)

    model = store.get_model("saved_model.pkl")
    df_test["score"] = model.predict(df_test)

    selected = choose_best_driver(df_test)
    store.put_predictions("results.csv", selected)

    print("🎉 Prediction complete!")


def choose_best_driver(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby("order_id").agg({"driver_id": list, "score": list}).reset_index()

    df["best_driver"] = df.apply(
        lambda r: r["driver_id"][np.argmax(r["score"])], axis=1
    )

    return df[["order_id", "best_driver"]].rename(columns={"best_driver": "driver_id"})


if __name__ == "__main__":
    main()
