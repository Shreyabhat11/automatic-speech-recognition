import os
import pandas as pd

from jiwer import wer, cer

from models.whisper_local import transcribe_whisper
from models.deepgram_api import transcribe_deepgram


GROUND_TRUTH_CSV = "data/transcripts/ground_truths.csv"
AUDIO_DIR = "data/processed_audio"

OUTPUT_CSV = "evaluation/benchmark_results.csv"

os.makedirs("evaluation", exist_ok=True)

df = pd.read_csv(GROUND_TRUTH_CSV)

results = []

print("\nStarting Benchmark...\n")

for idx, row in df.iterrows():

    filename = row["filename"]
    locality = row["locality"]
    ground_truth = row["ground_truth"]
    condition = row["condition"]

    audio_path = os.path.join(AUDIO_DIR, filename)

    print(f"\nProcessing: {filename}")

    # ---------------------------
    # WHISPER
    # ---------------------------

    try:

        whisper_result = transcribe_whisper(audio_path)

        whisper_transcript = whisper_result["transcript"]
        whisper_latency = whisper_result["latency"]

        whisper_wer = wer(
            ground_truth.lower(),
            whisper_transcript.lower()
        )

        whisper_cer = cer(
            ground_truth.lower(),
            whisper_transcript.lower()
        )

        whisper_entity = locality.lower() in whisper_transcript.lower()

        results.append({
            "filename": filename,
            "condition": condition,
            "model": "whisper",
            "locality": locality,
            "ground_truth": ground_truth,
            "transcript": whisper_transcript,
            "wer": round(whisper_wer, 3),
            "cer": round(whisper_cer, 3),
            "entity_correct": whisper_entity,
            "latency": whisper_latency
        })

        print(f"Whisper Done")

    except Exception as e:

        print(f"Whisper failed: {e}")

    # ---------------------------
    # DEEPGRAM
    # ---------------------------

    try:

        deepgram_result = transcribe_deepgram(audio_path)

        deepgram_transcript = deepgram_result["transcript"]
        deepgram_latency = deepgram_result["latency"]

        deepgram_wer = wer(
            ground_truth.lower(),
            deepgram_transcript.lower()
        )

        deepgram_cer = cer(
            ground_truth.lower(),
            deepgram_transcript.lower()
        )

        deepgram_entity = locality.lower() in deepgram_transcript.lower()

        results.append({
            "filename": filename,
            "condition": condition,
            "model": "deepgram",
            "locality": locality,
            "ground_truth": ground_truth,
            "transcript": deepgram_transcript,
            "wer": round(deepgram_wer, 3),
            "cer": round(deepgram_cer, 3),
            "entity_correct": deepgram_entity,
            "latency": deepgram_latency
        })

        print(f"Deepgram Done")

    except Exception as e:

        print(f"Deepgram failed: {e}")

# --------------------------------
# SAVE RESULTS
# --------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_CSV, index=False)

print("\nBenchmark completed.")
print(f"Results saved to: {OUTPUT_CSV}")