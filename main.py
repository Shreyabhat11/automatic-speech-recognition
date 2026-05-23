import pandas as pd
from tqdm import tqdm

from models.whisper_local import transcribe_whisper
from models.deepgram_api import transcribe_deepgram

from evaluation.metrics import calculate_metrics
from evaluation.entity_eval import locality_detected

df = pd.read_csv("data/transcripts/ground_truth.csv")

results = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    audio_path = f"data/raw_audio/{row['filename']}"

    ground_truth = row["ground_truth"]
    locality = row["locality"]

    whisper_result = transcribe_whisper(audio_path)
    deepgram_result = transcribe_deepgram(audio_path)

    whisper_metrics = calculate_metrics(
        ground_truth,
        whisper_result["transcript"]
    )

    deepgram_metrics = calculate_metrics(
        ground_truth,
        deepgram_result["transcript"]
    )

    results.append({
        "filename": row["filename"],

        "whisper_transcript": whisper_result["transcript"],
        "deepgram_transcript": deepgram_result["transcript"],

        "whisper_wer": whisper_metrics["WER"],
        "deepgram_wer": deepgram_metrics["WER"],

        "whisper_cer": whisper_metrics["CER"],
        "deepgram_cer": deepgram_metrics["CER"],

        "whisper_entity_correct":
            locality_detected(locality, whisper_result["transcript"]),

        "deepgram_entity_correct":
            locality_detected(locality, deepgram_result["transcript"]),

        "whisper_latency": whisper_result["latency"],
        "deepgram_latency": deepgram_result["latency"],

        "condition": row["condition"]
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    "reports/benchmark_results.csv",
    index=False
)

print(results_df.head())