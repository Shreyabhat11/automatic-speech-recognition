import os
import time
import whisper

# Explicit FFmpeg path
FFMPEG_PATH = r"C:\Users\shreyabhat\ffmpeg\bin"

# Add ffmpeg to environment path for this process
os.environ["PATH"] += os.pathsep + FFMPEG_PATH

print("Loading Whisper model...")

model = whisper.load_model("tiny")

print("Whisper model loaded.")


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