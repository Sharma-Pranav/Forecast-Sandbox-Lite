from typing import Dict
import warnings
from pathlib import Path
import pandas as pd
from typing import Any
from tqdm import tqdm
import numpy as np
from utils.metrics import mae, bias
from statsforecast import StatsForecast
from statsforecast.models import (
    # Baselines
    Naive,
    SeasonalNaive,
    RandomWalkWithDrift,
    HistoricAverage,
    WindowAverage,

    # Exponential smoothing / Holt / Holt-Winters
    SimpleExponentialSmoothingOptimized,
    SeasonalExponentialSmoothingOptimized,
    Holt,
    HoltWinters,

    # Theta family
    Theta,
    OptimizedTheta,
    DynamicTheta,
    DynamicOptimizedTheta,

    # ARIMA
    # AutoARIMA,

    # Intermittent
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
)


### ----- Configuration ----- ###
TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")
METRICS_PATH = Path("metrics/baseline_metrics.csv")    
PREDICTIONS_PATH = Path("metrics/baseline_predictions.csv")
### ------------------------- ###

HORIZON = 14  # Forecast horizon
warnings.filterwarnings("ignore")


# Adding type hints for better code clarity and numpy style comments for documentation  
def build_baseline_models(
        season_length: int = 7,
        window_size: int = 4,
    ) -> Dict[str, Any]:
    """Build a dictionary of baseline forecasting models.
    Parameters
    ----------
    season_length : int, optional
        Seasonality length, by default 7
    window_size : int, optional
        Window size for moving average models, by default 4
    Returns
    -------
    Dict[str, Any]
        Dictionary of baseline forecasting models
    """

    models= {
        # ----------------------
        # 1) Naive family
        # ----------------------
        str(Naive().__class__.__name__): Naive(),
        str(SeasonalNaive(season_length=season_length).__class__.__name__): SeasonalNaive(season_length=season_length),
        str(RandomWalkWithDrift().__class__.__name__): RandomWalkWithDrift(),
        str(HistoricAverage().__class__.__name__): HistoricAverage(),
        str(WindowAverage(window_size=window_size).__class__.__name__): WindowAverage(window_size=window_size),
        # ----------------------
        # 2) SES / Holt / Holt-Winters
        # ----------------------
        # SES ~ simple exponential smoothing
        str(SimpleExponentialSmoothingOptimized().__class__.__name__): SimpleExponentialSmoothingOptimized(),
        str(SeasonalExponentialSmoothingOptimized(season_length=season_length).__class__.__name__): SeasonalExponentialSmoothingOptimized(season_length=season_length),
        # Holt: level + trend, no seasonality
        str(Holt().__class__.__name__): Holt(), 
        str(HoltWinters(
            season_length=7,          # e.g. weekly seasonality for daily data # or "multiplicative"
        ).__class__.__name__): HoltWinters(
            season_length=7,          # e.g. weekly seasonality for daily data
        ),
        # ----------------------
        # 3) Theta family
        # ----------------------
        str(Theta().__class__.__name__): Theta(),
        str(OptimizedTheta().__class__.__name__): OptimizedTheta(),
        str(DynamicTheta().__class__.__name__): DynamicTheta(),
        str(DynamicOptimizedTheta().__class__.__name__): DynamicOptimizedTheta(),

        # ----------------------
        # 4) ARIMA baseline
        # ----------------------
        # str(AutoARIMA().__class__.__name__): AutoARIMA(season_length=season_length),

        # ----------------------
        # 5) Intermittent demand
        # ----------------------
        str(CrostonClassic().__class__.__name__): CrostonClassic(),
        str(CrostonSBA().__class__.__name__): CrostonSBA(),
        str(CrostonOptimized().__class__.__name__): CrostonOptimized(),

    }


    return models

# Adding type hints for better code clarity and numpy style comments for documentation  
def compute_baseline_forecasts(
    df: pd.DataFrame,
    models: Dict[str, Any],
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    df: train dataframe with columns ['id', 'date', 'sales']
    returns: long dataframe with columns
        ['id', 'model', 'h', 'forecast']
    """
    results = []

    sku_ids = df['id'].unique()
    for sku_id in tqdm(sku_ids):
        sku_data = df[df['id'] == sku_id].sort_values('date').copy()
        if len(sku_data) <= horizon + 5:
            continue

        sku_data.rename(columns={'sales': 'y', 'id': 'unique_id'}, inplace=True)
        # dummy calendar for StatsForecast — only order matters
        sku_data['ds'] = pd.date_range(start='2021-01-01', periods=len(sku_data), freq='D')
        sku_data = sku_data[['unique_id', 'ds', 'y']]

        for model_name, model in models.items():
            sf = StatsForecast(models=[model], freq='D', n_jobs=1)
            sf.fit(sku_data)
            forecast_df = sf.predict(h=horizon)

            # StatsForecast sometimes uses short aliases for columns
            # map model_name → column name
            col_map = {
                "RandomWalkWithDrift": "RWD",
                "SimpleExponentialSmoothingOptimized": "SESOpt",
                "SeasonalExponentialSmoothingOptimized": "SeasESOpt",
            }

            col = col_map.get(model_name, model.__class__.__name__)
            if col not in forecast_df.columns:
                raise ValueError(f"Column {col} not found for model {model_name}")

            forecast_values = forecast_df[col].values

            for step in range(horizon):
                results.append({
                    "id": sku_id,
                    "model": model_name,
                    "h": step + 1,
                    "forecast": float(forecast_values[step]),
                })

    return pd.DataFrame(results)
def compute_metrics(
    test_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    test_df: ['id', 'date', 'sales', ...]
    forecasts_df: ['id', 'model', 'h', 'forecast']
    returns: ['id', 'model', 'mae', 'bias', 'score']
    """
    test_df = test_df.sort_values(["id", "date"]).copy()

    # assign step index 1..H per SKU in time order
    test_df["h"] = test_df.groupby("id").cumcount() + 1

    metrics_rows = []
    for (sku_id, model_name), g_fore in forecasts_df.groupby(["id", "model"]):
        g_test = test_df[test_df["id"] == sku_id].copy()
        if g_test["h"].max() < horizon:
            # test shorter than horizon for some reason
            continue

        # align on h
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

        metrics_rows.append({
            "id": sku_id,
            "model": model_name,
            "mae": float(m),
            "bias": float(b),
            "score": float(s),
        })

    return pd.DataFrame(metrics_rows)


if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    # Build baseline models
    baseline_models = build_baseline_models()

    # Forecast from train into test horizon
    train_forecasts = compute_baseline_forecasts(train_df, baseline_models, horizon=HORIZON)

    # Save raw forecasts (for later UI)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_forecasts.to_csv(PREDICTIONS_PATH, index=False)

    # Compute metrics vs test
    metrics_df = compute_metrics(test_df, train_forecasts, horizon=HORIZON)
    metrics_df.to_csv(METRICS_PATH, index=False)

    print("Saved:")
    print(f" - forecasts → {PREDICTIONS_PATH}")
    print(f" - metrics   → {METRICS_PATH}")
    
