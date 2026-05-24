import os
import time
import whisper

FFMPEG_PATH = r"C:\Users\shreyabhat\ffmpeg\bin"

os.environ["PATH"] += os.pathsep + FFMPEG_PATH

print("Loading Whisper Tiny model...")

model = whisper.load_model("tiny")

print("Whisper Tiny loaded.")


def transcribe_whisper_tiny(audio_path):

    start = time.time()

    result = model.transcribe(
        audio_path,
        language="hi",
        fp16=False
    )

    end = time.time()

    return {
        "transcript": result["text"].strip(),
        "latency": round(end - start, 2)
    }