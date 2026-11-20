import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DATA_DIR = APP_DIR / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"

APP_TITLE = "Clinical Note Recorder API"
APP_DESCRIPTION = "Local recording storage and playback endpoints."
MAX_RECORDING_BYTES = 100 * 1024 * 1024
MAX_TRANSCRIPT_LENGTH = 50_000
CHUNK_SIZE = 1024 * 1024
WHISPER_MODEL_SIZE = "small"
LIVE_WHISPER_MODEL_SIZE = "tiny"
WHISPER_CPU_THREADS = os.cpu_count() or 4
TRANSCRIPTION_LANGUAGE_CODES = frozenset(
    {"ar", "de", "en", "es", "fr", "hi", "it", "ja", "pt", "zh"}
)
LIVE_PROMPT_CONTEXT_WORDS = 48
LIVE_MEDICAL_PROMPT = (
    "Clinical note transcription. Common terms include patient, chief complaint, "
    "history of present illness, medical history, examination, assessment, "
    "diagnosis, treatment plan, medication, dosage, allergy, blood pressure, "
    "heart rate, respiratory rate, hypertension, diabetes, asthma, infection, "
    "follow-up, prescription, and laboratory results."
)
LIVE_CHUNK_SECONDS = 2.0
LIVE_OVERLAP_SECONDS = 0.5
LIVE_UNSTABLE_TAIL_SECONDS = 0.25

ALLOWED_AUDIO_TYPES = {
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/wav": "wav",
    "audio/webm": "webm",
    "audio/x-m4a": "m4a",
    "audio/x-wav": "wav",
}
