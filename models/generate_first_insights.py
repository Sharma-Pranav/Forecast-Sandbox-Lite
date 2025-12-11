# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("metrics/best_models.csv")

# global win rate
print(df["best_model"].value_counts(normalize=True).round(2))


# %%
full = pd.read_csv("data/processed/train.csv")

vol = full.groupby("id")["sales"].agg(["mean","std"]).reset_index()
vol["cv"] = vol["std"] / (vol["mean"] + 1e-9)

best = df.merge(vol[["id","cv"]], on="id")

best["cv_bin"] = pd.qcut(best["cv"], 3, labels=["Low","Mid","High"])
print(best.groupby(["cv_bin","best_model"]).size())

# %%
df = pd.read_csv("data/processed/train.csv")

g = df.groupby("id")["sales"]
summary = g.agg(["mean","std","count"])
summary = summary.rename(columns={"count":"T"})

summary["N"] = g.apply(lambda x: (x>0).sum())
summary["ADI"] = summary["T"] / summary["N"].replace(0,1)
summary["CV2"] = (summary["std"]/summary["mean"].replace(0,1))**2

summary.to_csv("metrics/demand_profile.csv")


# %%
summary["ADI_class"] = np.where(summary["ADI"] > 1.32, "High", "Low")
summary["CV2_class"] = np.where(summary["CV2"] > 0.49, "High", "Low")

summary["regime"] = summary["ADI_class"] + "-" + summary["CV2_class"]

# %%
best = pd.read_csv("metrics/best_models.csv")
merged = best.merge(summary[["ADI","CV2","regime"]], on="id", how="left")

# %%
merged.groupby("regime")["best_model"].value_counts(normalize=True).to_csv("metrics/regime_model_performance.csv")

# %%
merged.groupby('best_model').size()

# %%
merged.groupby('regime')['best_model'].value_counts()

# %%
merged.to_csv('metrics/best_by_sku.csv')

# %% [markdown]
# ##  Key Insight


# %%
print(best["best_model"].value_counts(normalize=True).round(2))

# %%
m = pd.read_csv("metrics/combined_metrics.csv")
print(m.groupby("model")["score"].mean().sort_values())

# %%
best[['id','best_model']]

import matplotlib.pyplot as plt
import numpy as np

# model_scores: Series indexed by model name, values = mean score
model_scores = m.groupby("model")["score"].mean().sort_values()

fig, ax = plt.subplots(figsize=(16, 10))

# -------------------------------
# Identify winner + tier bounds
# -------------------------------
best_model = model_scores.index[0]
best_value = model_scores.iloc[0]

# Tier boundaries (keep tunable)
tierB_upper = 70   # top band of "stable enough"
tierC_upper = 80   # start of "avoid if possible"

colors = []
for model, score in model_scores.items():
    if model == best_model:
        colors.append("#0047AB")        # Deep Royal Blue → portfolio winner
    elif score < tierB_upper:
        colors.append("#888888")        # Neutral Grey → stable / acceptable
    else:
        colors.append("#C43131")        # Executive Red → high-noise tier

bars = ax.bar(model_scores.index, model_scores.values, color=colors)

# -------------------------------
# Threshold reference lines
# -------------------------------
ax.axhline(tierB_upper, color="#666666", linestyle="--", linewidth=1)
ax.text(
    len(model_scores) - 0.3,
    tierB_upper + 0.8,
    "Stable Signal Threshold",
    color="#444444",
    fontsize=10,
    ha="right"
)

ax.axhline(tierC_upper, color="#444444", linestyle=":", linewidth=1)
ax.text(
    len(model_scores) - 0.3,
    tierC_upper + 0.8,
    "High-Noise Zone (Avoid for planning)",
    color="#444444",
    fontsize=10,
    ha="right"
)

# -------------------------------
# Value annotations
# -------------------------------
for bar, value in zip(bars, model_scores.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.8,
        f"{value:.2f}",
        ha="center",
        fontsize=9,
        fontweight="bold" if value == best_value else "normal"
    )

# -------------------------------
# Styling
# -------------------------------
ax.set_title(
    "Portfolio-Level Model Performance\n(Lower Score = More Stable, Lower Error Risk)",
    fontsize=18,
    fontweight="bold",
    pad=16,
)

ax.set_ylabel("Mean Score (MAE + |Bias|)", fontsize=12)
ax.set_xlabel("Forecasting Models", fontsize=12)

# Fix tick warning: set ticks explicitly, then labels
x_positions = np.arange(len(model_scores))
ax.set_xticks(x_positions)
ax.set_xticklabels(model_scores.index, rotation=45, ha="right")

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig("docs/model_score_ranking.png", dpi=300, bbox_inches="tight")
plt.show()











# Tier boundaries (tunable later)
tierB_upper = 70
tierC_upper = 80

# -------------------------------------------------
# PREP
# -------------------------------------------------
model_scores = m.groupby("model")["score"].mean().sort_values()

best_model = model_scores.index[0]
best_value = model_scores.iloc[0]

# Executive tiers


# Executive-friendly group labels
stable_models = model_scores[model_scores < tierB_upper].index
secondary_models = model_scores[(model_scores >= tierB_upper) & (model_scores < tierC_upper)].index
high_noise_models = model_scores[model_scores >= tierC_upper].index

def group_color(model):
    if model == best_model:
        return "#0047AB"             # Royal blue highlight
    elif model in stable_models:
        return "#4C8BBF"             # Soft blue-grey (Stable)
    elif model in secondary_models:
        return "#A5A5A5"             # Neutral grey (Secondary)
    else:
        return "#C43131"             # Executive red (High noise)

colors = [group_color(m) for m in model_scores.index]

# -------------------------------------------------
# PLOT
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))

bars = ax.bar(model_scores.index, model_scores.values, color=colors)

# -------------------------------------------------
# Threshold lines with simpler executive labels
# -------------------------------------------------
ax.axhline(tierB_upper, color="#666666", linestyle="--", linewidth=1)
ax.text(
    -0.3, tierB_upper + 0.5,
    "Stable Signal Threshold",
    fontsize=10, color="#444444", ha="left"
)

ax.axhline(tierC_upper, color="#888888", linestyle=":", linewidth=1)
ax.text(
    -0.3, tierC_upper + 0.5,
    "High-Noise Zone",
    fontsize=10, color="#444444", ha="left"
)

# -------------------------------------------------
# SIMPLIFIED VALUE LABELS
# -------------------------------------------------
for bar, value in zip(bars, model_scores.values):
    if value < tierB_upper:
        label_color = "#002147"
    elif value < tierC_upper:
        label_color = "#444444"
    else:
        label_color = "#7A0000"

    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.5,
        f"{value:.2f}",
        ha="center",
        fontsize=9,
        color=label_color
    )

# -------------------------------------------------
# EXECUTIVE STYLING
# -------------------------------------------------
ax.set_title(
    "Forecast Model Stability Comparison\n(lower = more stable, lower risk)",
    fontsize=18,
    fontweight="bold",
    pad=20
)

ax.set_ylabel("Stability Score (MAE + |Bias|)", fontsize=12)

# Remove clutter by hiding long model names
ax.set_xticks([])
ax.set_xlabel("Model Families (ranked left → right)", fontsize=12)

# Add grouped legend explanation
group_handles = [
    plt.Rectangle((0,0),1,1, color="#4C8BBF"),
    plt.Rectangle((0,0),1,1, color="#A5A5A5"),
    plt.Rectangle((0,0),1,1, color="#C43131"),
    plt.Rectangle((0,0),1,1, color="#0047AB"),
]

ax.legend(
    group_handles,
    ["Stable Models", "Secondary Models", "High-Noise Models", "Best Performer"],
    frameon=False,
    loc="upper left"
)

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig("docs/model_score_ranking_exec.png", dpi=300, bbox_inches="tight")
plt.show()
