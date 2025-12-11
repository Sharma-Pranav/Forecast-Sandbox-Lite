
---

# **Forecast Sandbox Lite — FreshNet Edition**

A reproducible sandbox for evaluating forecasting models on **fresh-retail SKU demand**, using a **bias-aware stability metric** and selecting the most deployable model per SKU.

Built on the **FreshRetailNet-50K** dataset — a large-scale, multi-store fresh retail forecasting corpus.

---

## **Dataset Reference — FreshRetailNet-50K (Hugging Face)**

This sandbox uses the public dataset:

**Dataset:** FreshRetailNet-50K
**Publisher:** Dingdong Inc.
**Hugging Face Link:**
[https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)

**Key dataset properties:**

* 50,000+ fresh SKU time series
* full product hierarchy: city → store → category → SKU
* sales, stockout hours, discount, promotion, temperature, precipitation, humidity, wind
* highly intermittent + volatile patterns typical of fresh operations
* multiple behavioral regimes (smooth, erratic, intermittent, lumpy)

### **How This Dataset Is Used in the Sandbox**

* A **stratified SKU subset** was sampled to preserve the **original ADI–CV² distribution**.
* Each sampled SKU retains **full historical trajectory**.
* **Sales values were scaled ×100** to remove decimal-range noise (e.g., 0.1 → 10).
* No raw dataset files are redistributed — only processed evaluation inputs.

To reproduce:
Users must download the dataset directly from Hugging Face.

---

## **What This Repository Provides**

This repository includes:

* classical models (SES, Holt, Holt-Winters, Theta)
* intermittent models (Croston variants)
* moving average baselines
* global ML model (LightGBM)
* **Chronos2 — foundation-model forecasting from Amazon**

Outputs include:

* SKU-wise metrics ledger
* complete model selection audit
* forecasts for all models
* Streamlit dashboard for exploration

This is not a model zoo; it is a **governance-driven forecasting pipeline**.

---

## **Evaluation Principle**

```text
Score = MAE + |Bias|
```

This metric penalizes unstable or directionally-wrong forecasts — a major cost driver in fresh retail.

---

## **Key Findings (FreshNet v1.0)**

### **1. Forecasting is regime-dependent.**

Different models dominate different ADI–CV² regimes.

### **2. Stability-first models win:**

Top performers across FreshRetailNet-50K:

* **DynamicOptimizedTheta**
* **SES / Holt / Holt-Winters**
* **SeasonalExpSmoothingOptimized**
* **Croston variants (SBA / Classic)**
* **Chronos2** — robust across mixed demand patterns
* LightGBM (only improves with covariates)

### **3. Foundation model (Chronos2) behaves correctly**

Strong across varied noise profiles, especially where continuity exists.

### **4. ML without drivers is expected to underperform**

FreshRetailNet-50K includes rich covariates (discount, weather, stockout signals).
When these are withheld by design, LightGBM produces minimal structure — correctly.

---

## **Repository Structure**

```text
/app                     → Streamlit UI
/data/processed          → subset + scaled FreshRetailNet-50K data
/metrics                 → forecasts, metrics, audit logs
/models                  → classical + Chronos2 wrappers
/utils                   → helpers
run_pipeline.py          → orchestrates the full pipeline
```

---

## **How to Run**

### Pipeline:

```bash
python run_pipeline.py
```

Outputs:

```
metrics/
    combined_metrics.csv
    best_models.csv
    model_selection_audit.csv
```

### Dashboard:

```bash
streamlit run app/app.py
```

---

## **Why Stability Matters in Fresh Retail**

Volatile daily demand causes frequent directional shifts.
Every shift forces:

* replanning
* overrides
* waste corrections
* order rebalancing

A stable forecast reduces resets and increases decision velocity.

---

## **Deployment**

Works on:

* **Hugging Face Spaces**
* **Docker**
* Local Streamlit execution

Fully self-contained.

---

## **Summary**

This sandbox delivers:

1. a **stable, SKU-specific recommended model**
2. a **bias-aware, regime-informed evaluation**
3. a **comprehensive audit log**
4. foundation-model benchmarking via **Chronos2**
5. a reproducible operational forecasting pipeline

Forecasting becomes **explainable, defendable, and operationally aligned** — one SKU at a time.

---

## **Version**

**FreshNet Release — v1.0 (December 2025)**

---
