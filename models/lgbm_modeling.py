import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from utils.metrics import mae, bias
from pathlib import Path

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
HORIZON = 14  # FreshRetailNet setup: 14-step forecast

TRAIN_DATA_PATH = Path("data/processed/lgbm_ready/train.csv")
TARGET_DATA_PATH = Path("data/processed/lgbm_ready/target.csv")
METRICS_PATH = Path("metrics/lgbm_metrics.csv")
PREDICTIONS_PATH = Path("metrics/lgbm_predictions.csv")

INFERENCE_PATH = Path("data/processed/lgbm_ready/inference/inference_train.csv")
VALIDATION_PATH = Path("data/processed/lgbm_ready/inference/inference_target.csv")

# ---------------------------------------------------------
# Base single-target LGBM
# ---------------------------------------------------------
base_lgbm = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=0,
)

model = MultiOutputRegressor(base_lgbm)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For LGBM multi-output here, each row is:
    id | t1 | t2 | ... | tN

    We drop 'id' and use the time columns as features/targets.
    """
    return df.drop(columns=["id"])


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
train_data = pd.read_csv(TRAIN_DATA_PATH)
target_data = pd.read_csv(TARGET_DATA_PATH)

inference_data = pd.read_csv(INFERENCE_PATH)
validation_data = pd.read_csv(VALIDATION_PATH)

# Sanity
# print(train_data.head())
# print(target_data.head())
# print(inference_data.head())
# print(validation_data.head())

train_df = build_features(train_data)
target_df = build_features(target_data)
inference_df = build_features(inference_data)
validation_df = build_features(validation_data)

# ---------------------------------------------------------
# Fit model
# ---------------------------------------------------------
model.fit(train_df, target_df)

# Predict for inference windows
preds = model.predict(inference_df)
# If you want integer-unit forecasts, uncomment:
# preds = np.round(preds)

# ---------------------------------------------------------
# Build long-format prediction DataFrame
# ---------------------------------------------------------
preds_df = pd.DataFrame(preds)  # shape: [n_rows, HORIZON]

# Long form: one row per (row_index, horizon_step)
preds_df_long = preds_df.stack().reset_index()
preds_df_long.columns = ["id_index", "h_raw", "forecast"]

# Map back to actual ids using validation_data row order
preds_df_long["id"] = validation_data.iloc[preds_df_long["id_index"]]["id"].values
preds_df_long["h"] = preds_df_long["h_raw"].astype(int) + 1  # 1..HORIZON
preds_df_long["model"] = "lightgbm"

preds_df_final = preds_df_long[["id", "model", "h", "forecast"]]
print(preds_df_final.head(HORIZON))

# ---------------------------------------------------------
# Build long-format truth for validation horizon
# ---------------------------------------------------------
long = validation_data.melt(
    id_vars=["id"],
    var_name="h",
    value_name="sales",
)
long["h"] = long["h"].astype(int)

# ---------------------------------------------------------
# Compute metrics per id
# ---------------------------------------------------------
metrics_rows = []
for (sku_id, model_name), g_fore in preds_df_final.groupby(["id", "model"]):
    g_test = long[long["id"] == sku_id].copy()
    if g_test["h"].max() < HORIZON:
        # safety check – skip if somehow shorter
        continue

    merged = pd.merge(
        g_test[["id", "h", "sales"]],
        g_fore[["id", "h", "forecast"]],
        on=["id", "h"],
        how="inner",
    )
    if merged.empty:
        continue

    y_true = merged["sales"].values
    y_pred = merged["forecast"].values

    m = mae(y_true, y_pred)
    b = bias(y_true, y_pred)
    s = m + abs(b)

    metrics_rows.append(
        {
            "id": sku_id,
            "model": model_name,
            "mae": float(m),
            "bias": float(b),
            "score": float(s),
        }
    )

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(METRICS_PATH, index=False)
preds_df_final.to_csv(PREDICTIONS_PATH, index=False)

print("Saved:")
print(f" - forecasts → {PREDICTIONS_PATH}")
print(f" - metrics   → {METRICS_PATH}")
