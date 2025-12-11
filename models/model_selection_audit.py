import pandas as pd

# Load winner table
ledger = pd.read_csv("metrics/best_by_sku.csv")  # IF already saved, else skip this line

# Load full metrics output (each model x sku)
metrics = pd.read_csv("metrics/combined_metrics.csv")


# Step 3: Rank models (in case needed later)
ledger["model_rank"] = ledger.groupby("id")["score"].rank(method="first")

# Step 4: Rename columns as required
ledger.rename(columns={
    "id": "sku",
    "model": "best_model"
}, inplace=True)

# Optional note column placeholder
ledger["note"] = ""

# Step 5: Export
ledger = ledger[["sku", "regime", "best_model", "mae", "bias", "score", "model_rank", "note"]]

ledger.to_csv("metrics/model_selection_audit.csv", index=False)

print("Ledger created at metrics/model_selection_audit.csv")
ledger.head()
