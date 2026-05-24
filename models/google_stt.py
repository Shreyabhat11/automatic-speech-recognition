import os
import time

from google.cloud import speech


# -----------------------------------
# GOOGLE CREDENTIALS
# -----------------------------------

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    r"D:\project\automatic-speech-recognition\gen-lang-client-0550490215-feca78951f8f.json"
)

# -----------------------------------
# CLIENT
# -----------------------------------

client = speech.SpeechClient()


def transcribe_google(audio_path):

    start = time.time()

    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(
        content=content
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="hi-IN",
        enable_automatic_punctuation=True
    )

    response = client.recognize(
        config=config,
        audio=audio
    )

    end = time.time()

    transcript = ""

    for result in response.results:
        transcript += result.alternatives[0].transcript + " "

    return {
        "transcript": transcript.strip(),
        "latency": round(end - start, 2)
    }