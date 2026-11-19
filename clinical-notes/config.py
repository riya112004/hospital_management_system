from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DATA_DIR = APP_DIR / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"

APP_TITLE = "Clinical Note Recorder API"
APP_DESCRIPTION = "Local recording storage and playback endpoints."
MAX_RECORDING_BYTES = 100 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
WHISPER_MODEL_SIZE = "small"

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
