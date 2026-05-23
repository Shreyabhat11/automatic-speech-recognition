from models.deepgram_api import transcribe_deepgram

audio = "data/processed_audio/001_HSR_layout_quiet.wav"

result = transcribe_deepgram(audio)

print(result)