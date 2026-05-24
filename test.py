from models.whisper_local import transcribe_whisper
from models.whisper_tiny import transcribe_whisper_tiny
from models.whisper_small import transcribe_whisper_small   
from models.whisper_large import transcribe_whisper_large
from models.deepgram_api import transcribe_deepgram
from models.google_stt import transcribe_google

audio = "data/processed_audio/001_hsr_layout_noise.wav"

result_whisper_base = transcribe_whisper(audio)
result_whisper_tiny = transcribe_whisper_tiny(audio)
result_whisper_small = transcribe_whisper_small(audio)
result_whisper_large = transcribe_whisper_large(audio)

result_deepgram = transcribe_deepgram(audio)    
result_google = transcribe_google(audio)


print(result_whisper_base)
print(result_whisper_tiny)
print(result_whisper_small)
print(result_whisper_large)
print(result_deepgram)
print(result_google)