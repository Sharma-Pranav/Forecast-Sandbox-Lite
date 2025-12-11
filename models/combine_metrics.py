import pandas as pd
import numpy as np
from pathlib import Path
LGBM_METRICS_PATH = Path("metrics/lgbm_metrics.csv")  
BASELINE_METRICS_PATH = Path("metrics/baseline_metrics.csv")
CHRONOS_METRICS_PATH = Path("metrics/chronos_metrics.csv")
COMBINED_METRICS_PATH = Path("metrics/combined_metrics.csv")

lgbm_metrics = pd.read_csv(LGBM_METRICS_PATH)
baseline_metrics = pd.read_csv(BASELINE_METRICS_PATH)
chronos_metrics = pd.read_csv(CHRONOS_METRICS_PATH)
combined_metrics = pd.concat([lgbm_metrics, baseline_metrics, chronos_metrics], ignore_index=True)
combined_metrics.to_csv(COMBINED_METRICS_PATH, index=False)
print(f"Saved combined metrics to {COMBINED_METRICS_PATH}")