import time
import torch

from nemo.collections.asr.models import ASRModel


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Parakeet ASR model...")

model = ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

model = model.to(device)

print("Parakeet model loaded.")


def transcribe_parakeet(audio_path):

    start = time.time()

    output = model.transcribe(
        [audio_path]
    )

    end = time.time()

    transcript = output[0]

    return {
        "transcript": transcript,
        "latency": round(end - start, 2)
    }