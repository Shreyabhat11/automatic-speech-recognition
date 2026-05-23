import os
import time

from dotenv import load_dotenv
from deepgram import DeepgramClient

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)


def transcribe_deepgram(audio_path):

    start = time.time()

    with open(audio_path, "rb") as audio:

        response = deepgram.listen.v1.media.transcribe_file(
            request=audio.read(),
            model="nova-2",
            language="hi",
            smart_format=True,
            punctuate=True,
        )

    end = time.time()

    transcript = (
        response.results.channels[0]
        .alternatives[0]
        .transcript
    )

    return {
        "transcript": transcript,
        "latency": round(end - start, 2)
    }