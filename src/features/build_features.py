import pandas as pd

from src.features.transformations import (
    driver_distance_to_pickup,
    hour_of_day,
    driver_historical_completed_bookings,
)
from src.utils.store import AssignmentStore


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all transformations in correct order."""
    return (
        df.pipe(driver_distance_to_pickup)
          .pipe(hour_of_day)
          .pipe(driver_historical_completed_bookings)
    )


def main():
    store = AssignmentStore()

    print("Loading dataset...")
    df = store.get_processed("dataset.csv")

    print(" Applying feature engineering...")
    df = apply_feature_engineering(df)

    print(" Saving transformed dataset...")
    store.put_processed("transformed_dataset.csv", df)

    print(" Feature engineering complete!")


if __name__ == "__main__":
    main()
    print("DONE RUNNING BUILD_FEATURES.PY")
