from models.google_stt import transcribe_google

audio = "data/processed_audio/001_hsr_layout_quiet.wav"

result = transcribe_google(audio)

print(result)