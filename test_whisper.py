from models.whisper_local import transcribe_whisper

audio = "data/processed_audio/001_HSR_layout_quiet.wav"

result = transcribe_whisper(audio)

print(result)