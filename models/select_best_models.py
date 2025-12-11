import pandas as pd
from pathlib import Path

METRICS_PATH = Path("metrics/combined_metrics.csv")
BEST_PATH = Path("metrics/best_models.csv")
BEST_MODEL_OVERALL_PATH = Path("metrics/best_model_overall.csv")


def main():
    df = pd.read_csv(METRICS_PATH)

    # Identify best model per SKU by minimum score
    best = (
        df.sort_values(["id", "score"])
          .groupby("id")
          .head(1)
          .reset_index(drop=True)
    )

    best = best.rename(columns={
        "model": "best_model"
    })

    best.to_csv(BEST_PATH, index=False)
    print(best["best_model"].value_counts(normalize=True).round(2))
    print(f"Saved best models to {BEST_PATH}")
    df.groupby("model")["score"].mean().sort_values().to_csv(BEST_MODEL_OVERALL_PATH)


if __name__ == "__main__":
    main()