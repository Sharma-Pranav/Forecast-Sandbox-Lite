import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

import streamlit as st
import plotly.graph_objects as go


# -------------------
# Paths
# -------------------
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
METRICS_DIR = BASE_DIR / "metrics"

TEST_PATH = DATA_DIR / "test.csv"
BEST_MODELS_PATH = METRICS_DIR / "best_models.csv"
COMBINED_METRICS_PATH = METRICS_DIR / "combined_metrics.csv"
BASELINE_PRED_PATH = METRICS_DIR / "baseline_predictions.csv"
LGBM_PRED_PATH = METRICS_DIR / "lgbm_predictions.csv"
CHRONOS_PRED_PATH = METRICS_DIR / "chronos_predictions.csv"
DEMAND_PROFILE_PATH = METRICS_DIR / "demand_profile.csv"  # ADI / CV2
BEST_MODEL_OVERALL_PATH = METRICS_DIR / "best_model_overall.csv"


# -------------------
# Cached loaders
# -------------------
@st.cache_data
def load_test() -> pd.DataFrame:
    df = pd.read_csv(TEST_PATH)
    # ensure date sorted & numeric if needed
    return df.sort_values(["id", "date"]).reset_index(drop=True)


@st.cache_data
def load_best_models() -> pd.DataFrame:
    return pd.read_csv(BEST_MODELS_PATH)


@st.cache_data
def load_best_model_overall() -> pd.DataFrame:
    return pd.read_csv(BEST_MODEL_OVERALL_PATH)


@st.cache_data
def load_combined_metrics() -> pd.DataFrame:
    return pd.read_csv(COMBINED_METRICS_PATH)


@st.cache_data
def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Baselines: metrics/baseline_predictions.csv
        columns: id, model, h, forecast

    LightGBM: metrics/lgbm_predictions.csv
        columns: id, h (optional), forecast (or similar)

    Chronos2: metrics/chronos_predictions.csv
        columns: id, h, q10/q50/q90 or 0.1/0.5/0.9 etc.
    """
    # Baseline family (classical / Croston / theta / etc.)
    df_base = pd.read_csv(BASELINE_PRED_PATH)

    # LightGBM
    df_lgbm = pd.read_csv(LGBM_PRED_PATH)
    df_lgbm["model"] = "lightgbm"

    # Chronos2
    df_chronos = pd.read_csv(CHRONOS_PRED_PATH)

    # Normalize Chronos forecast column → 'forecast'
    if "q50" in df_chronos.columns:
        df_chronos = df_chronos.rename(columns={"q50": "forecast"})
    elif "0.5" in df_chronos.columns:
        df_chronos = df_chronos.rename(columns={"0.5": "forecast"})
    elif "predictions" in df_chronos.columns:
        df_chronos = df_chronos.rename(columns={"predictions": "forecast"})

    # Ensure an 'h' column exists for horizon ordering
    if "h" not in df_chronos.columns:
        # if no explicit horizon, infer by group order
        df_chronos["h"] = df_chronos.groupby("id").cumcount() + 1

    return df_base, df_lgbm, df_chronos


@st.cache_data
def load_demand_profile() -> Optional[pd.DataFrame]:
    if DEMAND_PROFILE_PATH.exists():
        return pd.read_csv(DEMAND_PROFILE_PATH)
    return None


# -------------------
# Helper: align predictions to test dates
# -------------------
def align_with_test_dates(
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    sku_id: str,
    model_name: Optional[str] = None,
    horizon_col: str = "h",
) -> pd.DataFrame:
    """
    Align predictions for a given SKU (and optional model) to the dates in test_df.

    Logic:
    - Take all test rows for this SKU and sort by 'date'.
    - Take all prediction rows for this SKU (and model, if given).
    - For baselines/Chronos2: sort by horizon_col (e.g. 'h').
      For LightGBM: sort by existing 'date' or index (we ignore its date values).
    - Overwrite/add a 'date' column in predictions using the test dates (by position).
    """
    # 1) Test horizon for this SKU
    sku_test = test_df[test_df["id"] == sku_id].sort_values("date")
    dates = sku_test["date"].values

    # 2) Filter predictions
    sku_pred = pred_df.copy()
    if "id" in sku_pred.columns:
        sku_pred = sku_pred[sku_pred["id"] == sku_id].copy()
    if model_name is not None and "model" in sku_pred.columns:
        sku_pred = sku_pred[sku_pred["model"] == model_name].copy()

    if sku_pred.empty:
        return sku_pred

    # 3) Sort predictions by available structure
    if horizon_col in sku_pred.columns:
        # baselines / Chronos: use horizon 'h'
        sku_pred = sku_pred.sort_values(horizon_col)
    else:
        # LightGBM: ignore whatever 'date' means, just use row order
        if "date" in sku_pred.columns:
            sku_pred = sku_pred.sort_values("date")
        else:
            sku_pred = sku_pred.sort_index()

    sku_pred = sku_pred.reset_index(drop=True)

    # 4) Map dates 1:1 by position
    n = min(len(dates), len(sku_pred))
    sku_pred = sku_pred.iloc[:n].copy()
    sku_pred["date"] = dates[:n]

    return sku_pred


# -------------------
# Helper: classify regime (for display)
# -------------------
def classify_regime(row, adi_thr: float = 1.32, cv2_thr: float = 0.49) -> str:
    adi_class = "High" if row["ADI"] > adi_thr else "Low"
    cv2_class = "High" if row["CV2"] > cv2_thr else "Low"

    if adi_class == "Low" and cv2_class == "Low":
        return "Low-Low (Smooth)"
    if adi_class == "Low" and cv2_class == "High":
        return "Low-High (Erratic)"
    if adi_class == "High" and cv2_class == "Low":
        return "High-Low (Intermittent)"
    return "High-High (Lumpy)"


# -------------------
# Main app
# -------------------
def main() -> None:
    st.set_page_config(
        page_title="Forecast Sandbox Lite",
        layout="wide",
    )

    st.title("Forecast Sandbox Lite — SKU Explorer")
    st.caption("Interactive view of model selection, regime profile, and forecast vs actuals.")

    # ---- load core data ----
    test_df = load_test()
    best_df = load_best_models()
    best_model_overall = load_best_model_overall()
    metrics_df = load_combined_metrics()
    df_base, df_lgbm, df_chronos = load_predictions()
    demand_prof = load_demand_profile()

    skus = sorted(test_df["id"].unique())
    selected_sku = st.selectbox("Select SKU", skus)

    # ---- left: summary info ----
    col_info, col_plot = st.columns([1, 2])

    with col_info:
        st.subheader("SKU Summary")

        # best model
        row_best = best_df[best_df["id"] == selected_sku]
        if row_best.empty:
            st.error("No best model found for this SKU.")
            return

        best_model = row_best["best_model"].iloc[0]
        st.markdown(f"**Recommended Model:** `{best_model}`")

        # metrics for this SKU
        sku_metrics = (
            metrics_df[metrics_df["id"] == selected_sku]
            .sort_values("score")
            .reset_index(drop=True)
        )
        best_row_metric = sku_metrics[sku_metrics["model"] == best_model].iloc[0]

        st.markdown("**Model Performance (Score = MAE + |Bias|)**")
        st.write(
            {
                "Score": round(best_row_metric["score"], 3),
                "MAE": round(best_row_metric["mae"], 3),
                "Bias": round(best_row_metric["bias"], 3),
            }
        )

        # regime info (ADI / CV2) if available
        if (
            demand_prof is not None
            and "ADI" in demand_prof.columns
            and "CV2" in demand_prof.columns
        ):
            row_prof = demand_prof[demand_prof["id"] == selected_sku]
            if not row_prof.empty:
                row_prof = row_prof.iloc[0]
                regime_label = classify_regime(row_prof)
                st.markdown("**Demand Regime (ADI–CV²):**")
                st.write(
                    {
                        "ADI": round(row_prof["ADI"], 2),
                        "CV²": round(row_prof["CV2"], 2),
                        "Regime": regime_label,
                    }
                )

        st.markdown("---")
        st.markdown("**All Models for This SKU**")
        st.dataframe(
            sku_metrics[["model", "mae", "bias", "score"]],
            use_container_width=True,
            height=300,
        )

    # ---- right: plot ----
    with col_plot:
        st.subheader("Actual vs Forecast")

        sku_test = test_df[test_df["id"] == selected_sku].sort_values("date")

        # ---- align predictions with test dates ----
        if best_model == "lightgbm":
            raw_pred = df_lgbm
            sku_pred = align_with_test_dates(
                test_df=test_df,
                pred_df=raw_pred,
                sku_id=selected_sku,
                model_name=None,   # df_lgbm already only has lightgbm
                horizon_col="h",   # will be ignored if missing
            )
        elif best_model == "chronos2":
            raw_pred = df_chronos
            sku_pred = align_with_test_dates(
                test_df=test_df,
                pred_df=raw_pred,
                sku_id=selected_sku,
                model_name=None,   # chronos df keyed only by id + h
                horizon_col="h",
            )
        else:
            # Baseline predictions for this SKU & best model
            raw_pred = df_base
            sku_pred = align_with_test_dates(
                test_df=test_df,
                pred_df=raw_pred,
                sku_id=selected_sku,
                model_name=best_model,
                horizon_col="h",
            )

        if sku_pred.empty:
            st.error("No predictions found for this SKU/model combination.")
            return

        # ensure a 'forecast' column exists
        if "forecast" not in sku_pred.columns:
            for cand in ["y_pred", "prediction", "pred", "yhat"]:
                if cand in sku_pred.columns:
                    sku_pred = sku_pred.rename(columns={cand: "forecast"})
                    break

        if "forecast" not in sku_pred.columns:
            st.error("Predictions for this SKU do not contain a 'forecast' column.")
            return

        sku_pred = sku_pred.sort_values("date")

        # merge actual + forecast on aligned 'date'
        merged = sku_test.merge(
            sku_pred[["date", "forecast"]],
            on="date",
            how="left",
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=merged["date"],
                y=merged["sales"],
                mode="lines",
                name="Actual",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=merged["date"],
                y=merged["forecast"],
                mode="lines+markers",
                name=f"Forecast ({best_model})",
            )
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        # download section
        st.markdown("### Download Forecast Data")
        csv = merged.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV for this SKU",
            data=csv,
            file_name=f"{selected_sku}_forecast_vs_actual.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
