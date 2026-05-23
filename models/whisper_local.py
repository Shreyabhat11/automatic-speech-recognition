import whisper
import time

model = whisper.load_model("large-v3")

def transcribe_whisper(audio_path):
    start = time.time()

    result = model.transcribe(audio_path)

    end = time.time()

    return {
        "transcript": result["text"],
        "latency": end - start
    }