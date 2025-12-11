#!/usr/bin/env python
"""
Forecast Sandbox v1.0
End-to-end pipeline runner.

Runs the full sequence:

1. python -m data.prepare_data
2. python -m data.prepare_data_lgbm
3. python -m models.compute_baselines
4. python -m models.lgbm_modeling
5. python -m models.combine_metrics
6. python -m models.select_best_models
7. python -m models.generate_first_insights
8. python -m models.model_selection_audit

## SCRIPT EXECUTION DEPENDENCIES GRAPH

```
prepare_data ──▶ prepare_data_lgbm ──▶ lgbm_modeling
              └─▶ compute_baselines
compute_baselines + lgbm_modeling ──▶ combine_metrics
combine_metrics ──▶ select_best_model
select_best_model ──▶ generate_first_insights
generate_first_insights ──▶ model_selection_audit
```

"""

import subprocess
import sys
from datetime import datetime

STEPS = [
    # ("Prepare FreshNet-50K-derived train/test data", ["python", "-m", "data.prepare_data_freshnet"]),
    # ("Prepare LightGBM training dataset", ["python", "-m", "data.prepare_data_lgbm_fresh"]),
    # ("Compute statistical baselines", ["python", "-m", "models.compute_baselines"]),
    # ("Train LightGBM model", ["python", "-m", "models.lgbm_modeling"]),
    ("Inference on Chronos2 model", ["python", "-m", "models.chronos_inference"]),
    ("Combine metrics", ["python", "-m", "models.combine_metrics"]),
    ("Select best model per SKU", ["python", "-m", "models.select_best_models"]),
    ("Generate first insights & plots", ["python", "-m", "models.generate_first_insights"]),
    ("Create model selection audit ledger", ["python", "-m", "models.model_selection_audit"]),
]


def run_step(description: str, cmd: list[str]) -> None:
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] ▶ {description}")
    print(" Command:", " ".join(cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        print(f"[ERROR] Step failed: {description}", file=sys.stderr)
        sys.exit(completed.returncode)
    print(f"[OK] {description}")


def main() -> None:
    print("=== Forecast Sandbox v1.0 — Full Pipeline Run ===")
    print(f"Started at: {datetime.now().isoformat(timespec='seconds')}\n")

    for description, cmd in STEPS:
        run_step(description, cmd)

    print(f"\nFinished at: {datetime.now().isoformat(timespec='seconds')}")
    print("All steps completed successfully.")
    print("Key outputs:")
    print("  - metrics/combined_metrics.csv")
    print("  - metrics/best_by_sku.csv")
    print("  - metrics/model_selection_audit.csv")
    print("  - insights/model_score_ranking.png")
    print("  - insights/regime_model_performance.csv")


if __name__ == "__main__":
    main()
