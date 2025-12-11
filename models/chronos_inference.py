import pandas as pd
import numpy as np
from chronos import Chronos2Pipeline
from pathlib import Path
from utils.metrics import mae, bias

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
HORIZON = 14

INFERENCE_INPUT_PATH = Path("data/processed/lgbm_ready/inference/inference_train.csv")
INFERENCE_TARGET_PATH = Path("data/processed/lgbm_ready/inference/inference_target.csv")

CHRONOS_OUTPUT_PATH = Path("metrics/chronos_predictions.csv")
CHRONOS_METRICS_PATH = Path("metrics/chronos_metrics.csv")

# ---------------------------------------------------------
# LOAD CHRONOS MODEL
# ---------------------------------------------------------
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",  # you can switch to "cuda" once your env is ready
)

# ---------------------------------------------------------
# LOAD DATA (wide: id, 1..T)
# ---------------------------------------------------------
df_infer = pd.read_csv(INFERENCE_INPUT_PATH)
df_true = pd.read_csv(INFERENCE_TARGET_PATH)

# ---------------------------------------------------------
# Helper: wide -> Chronos long format
# ---------------------------------------------------------
def reshape_for_chronos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide format to Chronos format:
        id | timestamp | target
    We create a synthetic daily calendar starting at 2024-01-01.
    """
    melted = df.melt(id_vars=["id"], var_name="day", value_name="target")
    melted["day"] = melted["day"].astype(int)
    melted["timestamp"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        melted["day"], unit="D"
    )
    return melted[["id", "timestamp", "target"]]

context_df = reshape_for_chronos(df_infer)
truth_df = reshape_for_chronos(df_true)

# ---------------------------------------------------------
# CHRONOS INFERENCE (univariate, per-id)
# ---------------------------------------------------------
prediction_rows = []

for id_val, df_series in context_df.groupby("id"):
    df_series_sorted = df_series.sort_values("timestamp")

    pred_df = pipeline.predict_df(
        df_series_sorted,
        future_df=None,
        prediction_length=HORIZON,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    # Chronos returns columns like: timestamp, target_name, predictions, 0.1, 0.5, 0.9
    pred_df["id"] = id_val
    pred_df["h"] = np.arange(1, HORIZON + 1)  # horizon index

    prediction_rows.append(pred_df)

chronos_preds = pd.concat(prediction_rows, ignore_index=True)
print(chronos_preds.head())

# ---------------------------------------------------------
# Save predictions (rename quantile columns)
# ---------------------------------------------------------
chronos_preds.rename(
    columns={
        "0.1": "q10",
        "0.5": "q50",
        "0.9": "q90",
    },
    inplace=True,
)

chronos_preds.to_csv(CHRONOS_OUTPUT_PATH, index=False)
print(f"Saved Chronos forecasts → {CHRONOS_OUTPUT_PATH}")

# ---------------------------------------------------------
# METRICS — align truth with horizon index h
# ---------------------------------------------------------

# Take last HORIZON days per id from truth_df, ordered by timestamp
true_filtered = (
    truth_df.sort_values(["id", "timestamp"])
    .groupby("id", group_keys=False)
    .tail(HORIZON)
)

# Assign horizon index 1..H per id (aligned with Chronos h)
true_filtered["h"] = (
    true_filtered.sort_values(["id", "timestamp"])
    .groupby("id")
    .cumcount() + 1
)

print(true_filtered.head())

# Merge on id + h, use q50 as point forecast
merged = pd.merge(
    true_filtered[["id", "h", "target"]],
    chronos_preds[["id", "h", "q50"]],
    on=["id", "h"],
    how="inner",
)

merged.rename(columns={"q50": "forecast"}, inplace=True)

# ---------------------------------------------------------
# Compute metrics per id
# ---------------------------------------------------------
metric_rows = []

for id_val, g in merged.groupby("id"):
    y_true = g["target"].values
    y_pred = g["forecast"].values

    m = mae(y_true, y_pred)
    b = bias(y_true, y_pred)
    s = m + abs(b)

    metric_rows.append(
        {
            "id": id_val,
            "model": "chronos2",
            "mae": float(m),
            "bias": float(b),
            "score": float(s),
        }
    )

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(CHRONOS_METRICS_PATH, index=False)

print(f"Saved metrics → {CHRONOS_METRICS_PATH}")
