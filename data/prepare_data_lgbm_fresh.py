import pandas as pd
import numpy as np
import os

# -------------------------------------------------------------------
# Load train/test built from FreshRetailNet-50K
# (from your new prepare_freshretailnet_subset.py pipeline)
# -------------------------------------------------------------------
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

print("Train columns:", train_df.columns.tolist())

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename pivoted time-step columns so they start from 1 and increment
    by 1, regardless of original day_idx values.

    Input columns: ["id", t1, t2, ..., tN]  (t's are ints)
    Output cols:   ["id", 1, 2, ..., N]
    """
    other_cols = [c for c in df.columns if c != "id"]
    if not other_cols:
        return df

    min_val = min(other_cols)
    new_cols = [int(c - min_val + 1) for c in other_cols]

    df.columns = ["id"] + new_cols
    return df


# -------------------------------------------------------------------
# CONFIG — automatically adapt to train length per id
# -------------------------------------------------------------------
# we keep your original 14-step validation for LGBM
lgbm_val_length = 14

# infer per-id train series length from the first id
_first_id = train_df["id"].iloc[0]
_series_len_train = train_df[train_df["id"] == _first_id].shape[0]

# ensure it's consistent across all ids
assert (
    train_df.groupby("id").size().nunique() == 1
), "All ids in train_df must have same length."

lgbm_train_length = _series_len_train - lgbm_val_length
print(f"Detected train length per id: {_series_len_train}")
print(f"LGBM train length: {lgbm_train_length}, val length: {lgbm_val_length}")

# -------------------------------------------------------------------
# 1) BUILD TRAIN/TARGET FOR MULTI-OUTPUT LGBM
# -------------------------------------------------------------------
train_dfs = []
target_dfs = []

for unique_id in train_df["id"].unique():
    subset_df = train_df[train_df["id"] == unique_id].copy()

    # Drop non-time-series numeric features; keep only id, day_idx, sales
    drop_cols = [
        "product_id",
        "category_1",
        "category_2",
        "category_3",
        "store_id",
        "city_id",
        "management_group_id",
        "sale_hours",
        "sale_hour_ratio",
        "stock_hour6_22_cnt",
        "stockout_hours",
        "stockout_hour_ratio",
        "avail_hour_ratio",
        "discount",
        "holiday",
        "activity_flag",
        "precip",
        "temp",
        "humidity",
        "wind_level",
        "mean",
        "std",
        "T",
        "N",
        "ADI",
        "CV2",
        "ADI_class",
        "CV2_class",
        "regime",
    ]
    drop_cols = [c for c in drop_cols if c in subset_df.columns]
    subset_df = subset_df.drop(columns=drop_cols)

    # We rely on day_idx as the time axis (1..62)
    # sort just to be sure
    subset_df = subset_df.sort_values("day_idx")

    # Rolling windows in the training set
    for start in range(0, len(subset_df), lgbm_train_length + lgbm_val_length):
        train_slice = subset_df.iloc[start : start + lgbm_train_length]
        test_slice = subset_df.iloc[
            start + lgbm_train_length : start + lgbm_train_length + lgbm_val_length
        ]

        # skip incomplete windows
        if len(train_slice) < lgbm_train_length or len(test_slice) < lgbm_val_length:
            continue

        # Wide format for multi-output regression: id × time_steps
        train_wide = (
            train_slice.pivot(index="id", columns="day_idx", values="sales")
            .reset_index()
        )
        test_wide = (
            test_slice.pivot(index="id", columns="day_idx", values="sales")
            .reset_index()
        )

        train_wide = rename_columns(train_wide)
        test_wide = rename_columns(test_wide)

        train_dfs.append(train_wide)
        target_dfs.append(test_wide)

# Concatenate all windows
if not train_dfs or not target_dfs:
    raise RuntimeError("No valid LGBM windows were created from train_df.")

train_lgbm = pd.concat(train_dfs, ignore_index=True)
target_lgbm = pd.concat(target_dfs, ignore_index=True)

# Make directory if not exists
os.makedirs("data/processed/lgbm_ready", exist_ok=True)

# Save
train_lgbm.to_csv("data/processed/lgbm_ready/train.csv", index=False)
target_lgbm.to_csv("data/processed/lgbm_ready/target.csv", index=False)

print("Saved LGBM train/target to data/processed/lgbm_ready/")

# -------------------------------------------------------------------
# 2) BUILD INFERENCE TRAIN/TARGET (LAST WINDOW)
# -------------------------------------------------------------------
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

inference_dfs = []
validation_dfs = []

for unique_id in train_df["id"].unique():
    subset_train = train_df[train_df["id"] == unique_id].copy()
    subset_train = subset_train.sort_values("day_idx")

    # last lgbm_train_length points of the train series
    start_idx = subset_train.shape[0] - lgbm_train_length
    inference_slice = subset_train.iloc[start_idx : start_idx + lgbm_train_length]

    inference_wide = (
        inference_slice.pivot(index="id", columns="day_idx", values="sales")
        .reset_index()
    )
    inference_wide = rename_columns(inference_wide)

    # corresponding last lgbm_val_length points from test series
    subset_test = test_df[test_df["id"] == unique_id].copy()
    subset_test = subset_test.sort_values("day_idx")

    start_val = subset_test.shape[0] - lgbm_val_length
    validation_slice = subset_test.iloc[start_val : start_val + lgbm_val_length]

    validation_wide = (
        validation_slice.pivot(index="id", columns="day_idx", values="sales")
        .reset_index()
    )
    validation_wide = rename_columns(validation_wide)

    inference_dfs.append(inference_wide)
    validation_dfs.append(validation_wide)

inference_df = pd.concat(inference_dfs, ignore_index=True)
validation_df = pd.concat(validation_dfs, ignore_index=True)

os.makedirs("data/processed/lgbm_ready/inference", exist_ok=True)

inference_df.to_csv(
    "data/processed/lgbm_ready/inference/inference_train.csv",
    index=False,
)
validation_df.to_csv(
    "data/processed/lgbm_ready/inference/inference_target.csv",
    index=False,
)

print("Saved LGBM inference train/target to data/processed/lgbm_ready/inference/")
