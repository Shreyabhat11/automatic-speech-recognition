import os
import pandas as pd

from models.whisper_local import transcribe_whisper
from models.whisper_tiny import transcribe_whisper_tiny
from models.whisper_small import transcribe_whisper_small
from models.deepgram_api import transcribe_deepgram
from models.google_stt import transcribe_google

from evaluation.metrics import calculate_wer, calculate_cer
from evaluation.entity_utils import fuzzy_entity_match
from evaluation.summary_report import generate_summary

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# CONFIG
# ==========================================

AUDIO_DIR = "data/processed_audio"

metadata = pd.read_csv(
    "data/transcripts/ground_truths.csv"
)

os.makedirs(
    "results",
    exist_ok=True
)

results = []


# ==========================================
# MODEL REGISTRY
# ==========================================

MODELS = {
    "whisper_base": transcribe_whisper,
    "whisper_tiny": transcribe_whisper_tiny,
    "whisper_small": transcribe_whisper_small,
    "deepgram": transcribe_deepgram,
    "google_stt": transcribe_google
}


# ==========================================
# BENCHMARK LOOP
# ==========================================

for _, row in metadata.iterrows():

    filename = row["filename"]
    locality = row["locality"]
    condition = row["condition"]
    ground_truth = row["ground_truth"]

    audio_path = os.path.join(
        AUDIO_DIR,
        filename
    )

    print(f"\nProcessing: {filename}")

    for model_name, model_fn in MODELS.items():

        print(f"Running {model_name}...")

        try:

            result = model_fn(audio_path)

            entity_correct, entity_score = (
                fuzzy_entity_match(
                    locality,
                    result["transcript"]
                )
            )

            results.append({
                "filename": filename,
                "condition": condition,
                "model": model_name,
                "locality": locality,
                "ground_truth": ground_truth,
                "transcript": result["transcript"],
                "wer": calculate_wer(
                    ground_truth,
                    result["transcript"]
                ),
                "cer": calculate_cer(
                    ground_truth,
                    result["transcript"]
                ),
                "entity_correct": entity_correct,
                "entity_score": entity_score,
                "latency": result["latency"]
            })

        except Exception as e:

            print(f"ERROR in {model_name}: {e}")


# ==========================================
# SAVE RAW RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/benchmark_results.csv",
    index=False
)

print("\nBenchmark results saved.")


# ==========================================
# SUMMARY REPORT
# ==========================================

summary_df = generate_summary(
    results_df
)

summary_df.to_csv(
    "results/summary_report.csv",
    index=False
)

print("\nSummary Report:")
print(summary_df)


# ==========================================
# PLOTS
# ==========================================

sns.set_style("whitegrid")


# ------------------------------------------
# WER
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=results_df,
    x="model",
    y="wer",
    estimator="mean"
)

plt.title("Average WER by Model")

plt.savefig(
    "results/wer_comparison.png",
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------
# LATENCY
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=results_df,
    x="model",
    y="latency",
    estimator="mean"
)

plt.title("Average Latency by Model")

plt.savefig(
    "results/latency_comparison.png",
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------
# ENTITY ACCURACY
# ------------------------------------------

entity_df = (
    results_df.groupby("model")["entity_correct"]
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

plt.savefig(
    "results/entity_accuracy.png",
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------
# CONDITION ANALYSIS
# ------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
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


print("\nPlots generated.")
print("\nPipeline complete.")