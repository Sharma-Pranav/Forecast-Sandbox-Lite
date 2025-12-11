# %%
import pandas as pd
import numpy as np

np.random.seed(42)

# 1. Load HF dataset and convert to pandas
raw_df = pd.read_parquet("data/raw/FreshRetailNet-50K/train.parquet")
df_fresh_eval = pd.read_parquet("data/raw/FreshRetailNet-50K/eval.parquet")


# %%
# raw_df = raw_df.head(2000)
# 2. Rename for cleanliness
df = raw_df.rename(columns={
    "dt": "date",
    "sale_amount": "sales",
    "first_category_id": "category_1",
    "second_category_id": "category_2",
    "third_category_id": "category_3",
    "holiday_flag": "holiday",
    "precpt": "precip",
    "avg_temperature": "temp",
    "avg_humidity": "humidity",
    "avg_wind_level": "wind_level",
})

df["date"] = pd.to_datetime(df["date"])

# 3. Build a stable SKU-ID like you did for M5
#    here: city–store–product triple → id
df["id"] = "CID" + df["city_id"].astype(str) + "_SID" + df["store_id"].astype(str) + "_PID" + df["product_id"].astype(str) + "_MGID" + \
    df["management_group_id"].astype(str) + "_CAT1" + df["category_1"].astype(str) + "-CAT2" + df["category_2"].astype(str) + "-CAT3" + df["category_3"].astype(str)

# %%
def extract_daily_features(row):
    hs = np.array(row["hours_sale"], dtype=float)        # length 24
    st = np.array(row["hours_stock_status"], dtype=int)  # 1 = out-of-stock

    sale_hours = (hs > 0).sum()
    sale_hour_ratio = sale_hours / 24.0

    stockout_hours = st.sum()
    stockout_hour_ratio = stockout_hours / 24.0
    avail_hour_ratio = 1.0 - stockout_hour_ratio
    
    return pd.Series({
        "sale_hours": sale_hours,
        "sale_hour_ratio": sale_hour_ratio,
        "stockout_hours": stockout_hours,
        "stockout_hour_ratio": stockout_hour_ratio,
        "avail_hour_ratio": avail_hour_ratio,
    })

daily_feats = df.apply(extract_daily_features, axis=1)
df = pd.concat([df, daily_feats], axis=1)


# %%
tidy_df = df[['id',
    "date",
    "city_id", "store_id", "product_id",
    "management_group_id", "category_1", "category_2", "category_3",
    "sales",
    "sale_hours", "sale_hour_ratio",
    "stockout_hours", "stockout_hour_ratio", "avail_hour_ratio",
    "stock_hour6_22_cnt",
    "discount", "holiday", "activity_flag",
    "precip", "temp", "humidity", "wind_level",
]].copy()

# %%
g = tidy_df.groupby("id")["sales"]

summary = g.agg(["mean", "std", "count"])
summary = summary.rename(columns={"count": "T"})

summary["N"] = g.apply(lambda x: (x > 0).sum())
summary["ADI"] = summary["T"] / summary["N"].replace(0, 1)
summary["CV2"] = (summary["std"] / summary["mean"].replace(0, 1)) ** 2

summary["ADI_class"] = np.where(summary["ADI"] > 1.32, "High", "Low")
summary["CV2_class"] = np.where(summary["CV2"] > 0.49, "High", "Low")
summary["regime"] = summary["ADI_class"] + "-" + summary["CV2_class"]


# %%
tidy_df = tidy_df.merge(summary, on="id", how="left")

# %%
tidy_high_high = tidy_df[tidy_df["regime"] == "High-High"]
tidy_low_high  = tidy_df[tidy_df["regime"] == "Low-High"]
tidy_high_low   = tidy_df[tidy_df["regime"] == "High-Low"]
tidy_low_low   = tidy_df[tidy_df["regime"] == "Low-Low"]

def sample_by_regime(df_regime: pd.DataFrame, num_ids_needed: int) -> pd.DataFrame:
    """
    Sample num_ids_needed unique IDs from df_regime and return all their history.
    """
    concat_df = []
    for i, sku_id in enumerate(df_regime["id"].unique()):
        if i < num_ids_needed:
            concat_df.append(df_regime[df_regime["id"] == sku_id])
        else:
            break
    if not concat_df:
        return pd.DataFrame(columns=df_regime.columns)
    return pd.concat(concat_df, ignore_index=True)

multiples = 3
df_high_high_sampled = sample_by_regime(tidy_high_high, 3 * multiples)
df_low_high_sampled  = sample_by_regime(tidy_low_high, 15 * multiples)
df_high_low_sampled   = sample_by_regime(tidy_high_low, 10 * multiples)
df_low_low_sampled   = sample_by_regime(tidy_low_low, 81 * multiples)
tidy_subset = pd.concat(
    [df_high_high_sampled, df_low_high_sampled, df_high_low_sampled, df_low_low_sampled],
    ignore_index=True
)

print(tidy_subset["regime"].value_counts())
print(tidy_subset["regime"].value_counts(normalize=True))


# %%
# Sort properly
tidy_subset = tidy_subset.sort_values(["id", "date"])

# Per-SKU day index (1..T within each id)
tidy_subset["day_idx"] = (
    tidy_subset
    .groupby("id")["date"]
    .rank(method="first")
    .astype(int)
)


# %%
# Sort properly
tidy_subset = tidy_subset.sort_values(["id", "date"])

# Per-SKU day index (1..T within each id)
tidy_subset["day_idx"] = (
    tidy_subset
    .groupby("id")["date"]
    .rank(method="first")
    .astype(int)
)


# %%
# a=b

# %%
# For 90-day series: use first 76 days as train, last 14 as test
HORIZON = 14
TRAIN_HORIZON_END = 90-HORIZON   # 90 - 28


train_df = tidy_subset[tidy_subset["day_idx"] <= TRAIN_HORIZON_END]
test_df  = tidy_subset[tidy_subset["day_idx"] > TRAIN_HORIZON_END]

# Inference input for LGBM (last 200 days equivalent; here min(200, series_len))
# For 90-day series you might just use last 62 or so; here we take last 62:
inference_input_df_lgbm = tidy_subset[
    tidy_subset["day_idx"] > (TRAIN_HORIZON_END - 62)
]

# %%
train_df.head()

train_df["sales"] = train_df["sales"] * 100
test_df["sales"] = test_df["sales"] * 100
inference_input_df_lgbm["sales"] = inference_input_df_lgbm["sales"] * 100
# %%
import os
os.makedirs("data/processed", exist_ok=True)

tidy_subset.to_csv("data/processed/freshretailnet_subset.csv", index=False)
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)
inference_input_df_lgbm.to_csv(
    "data/processed/inference_input_df_lgbm.csv",
    index=False
)

# %%



