# from modules.splitters import AudioSplitter
#
# config = {
#     "input_dir": "data",
#     "output_dir": "chunks",
#     "chunk_ms": 3 * 60 * 1000,  # 3 minutos
#     "overlap_ms": 10_000,       # 10 segundos de solapamiento
#     "target_frame_rate": 16000, # útil para ASR
#     "normalize_dbfs": -16.0
# }
#
# splitter = AudioSplitter(config)
# summary = splitter.run()
#
# print(summary)

from modules.transcriber import FasterWhisperTranscriber

config = {
    "audio_path": "chunks/001_audio/chunk_003.wav",  # o cualquier ruta
    "language": "es",
    "model": "Systran/faster-whisper-large-v3",   # podés usar "large-v2" si querés algo más liviano
    "device": "cpu",       # "auto" o "cuda"
}

tr = FasterWhisperTranscriber(config)
texto = tr.transcribe()
print(texto)
