# Example override and freezes
# OVERRIDE_THRESHOLD_BY_REGIME = {
#     "High-High": 0.30,  # 30%
#     "Low-High":  0.25,  # 25%
#     "Low-Low":   0.20,  # 20%
# }

# FREEZE_HORIZON_BY_REGIME = {
#     "High-High": 14,  # minimal stabilizing window
#     "Low-High":   7,
#     "Low-Low":   28,
# }

MODEL_PRIORITY_BY_REGIME = {
    "High-High": [
        "lightgbm",
        "SeasonalExponentialSmoothingOptimized",
        "AutoARIMA",
    ],
    "Low-High": [
        "lightgbm",
        "HoltWinters",
        "SeasonalExponentialSmoothingOptimized",
    ],
    "Low-Low": [
        "HistoricAverage",
        "lightgbm",
        "SimpleExponentialSmoothingOptimized",
    ],
    # Optional explicit intermittent cluster if you later tag it
    "Intermittent": [
        "CrostonSBA",
        "CrostonOptimized",
        "HistoricAverage",
    ],
}
