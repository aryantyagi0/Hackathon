import pandas as pd
from haversine import haversine
from src.utils.time import robust_hour_of_iso_date


def driver_distance_to_pickup(df: pd.DataFrame) -> pd.DataFrame:
    if any(col not in df.columns for col in 
           ["driver_latitude", "driver_longitude", "pickup_latitude", "pickup_longitude"]):
        raise KeyError("Required location columns missing")

    df["driver_distance"] = df.apply(
        lambda r: haversine(
            (r["driver_latitude"], r["driver_longitude"]),
            (r["pickup_latitude"], r["pickup_longitude"]),
        ),
        axis=1,
    )
    return df


def hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    if "event_timestamp" not in df.columns:
        raise KeyError("event_timestamp column missing")

    df["event_hour"] = df["event_timestamp"].apply(robust_hour_of_iso_date)
    return df


def driver_historical_completed_bookings(df: pd.DataFrame) -> pd.DataFrame:
    required = ["driver_id", "is_completed", "event_timestamp"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"{col} column missing")

    df = df.copy()

    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], format="mixed")

    df = df.sort_values(["driver_id", "event_timestamp"])

    df["driver_completed_bookings"] = (
        df.groupby("driver_id")["is_completed"]
          .apply(lambda s: s.shift().fillna(0).cumsum())
          .reset_index(drop=True)
    )

    return df
