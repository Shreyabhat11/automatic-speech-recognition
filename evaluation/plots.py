import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "results/benchmark_results.csv"
)

sns.set_style("whitegrid")


# ==========================================
# WER COMPARISON
# ==========================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="model",
    y="wer",
    estimator="mean"
)

plt.title("Average WER by Model")
plt.ylabel("WER")

plt.savefig(
    "results/wer_comparison.png",
    bbox_inches="tight"
)

plt.close()


# ==========================================
# LATENCY COMPARISON
# ==========================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="model",
    y="latency",
    estimator="mean"
)

plt.title("Average Latency by Model")
plt.ylabel("Seconds")

plt.savefig(
    "results/latency_comparison.png",
    bbox_inches="tight"
)

plt.close()


# ==========================================
# ENTITY ACCURACY
# ==========================================

entity_df = (
    df.groupby("model")["entity_correct"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=entity_df,
    x="model",
    y="entity_correct"
)

plt.title("Entity Recognition Accuracy")
plt.ylabel("Accuracy")

plt.savefig(
    "results/entity_accuracy.png",
    bbox_inches="tight"
)

plt.close()


# ==========================================
# CONDITION ANALYSIS
# ==========================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="condition",
    y="wer",
    hue="model",
    estimator="mean"
)

plt.title("WER by Audio Condition")

plt.savefig(
    "results/condition_analysis.png",
    bbox_inches="tight"
)

plt.close()

print("Plots generated.")