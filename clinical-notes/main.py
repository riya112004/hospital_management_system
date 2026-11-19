from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
from threading import Lock
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydantic import BaseModel

from config import (
    ALLOWED_AUDIO_TYPES,
    APP_DESCRIPTION,
    APP_TITLE,
    CHUNK_SIZE,
    MAX_RECORDING_BYTES,
    RECORDINGS_DIR,
    STATIC_DIR,
    WHISPER_MODEL_SIZE,
)


EXTENSION_AUDIO_TYPES = {
    extension: content_type
    for content_type, extension in ALLOWED_AUDIO_TYPES.items()
}
TRANSCRIPTION_LOCK = Lock()
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RecordingOut(BaseModel):
    id: str
    created_at: datetime
    content_type: str
    size_bytes: int
    duration_seconds: float | None
    transcript: str | None
    language: str | None
    audio_url: str
    download_url: str


class TranscriptionOut(BaseModel):
    recording_id: str
    text: str
    language: str | None


def metadata_path(recording_id: str) -> Path:
    return RECORDINGS_DIR / f"{recording_id}.json"


def read_recording_metadata(audio_path: Path) -> dict:
    recording_id = audio_path.stem
    details_path = metadata_path(recording_id)

    if details_path.exists():
        with details_path.open("r", encoding="utf-8") as details_file:
            return json.load(details_file)

    return {
        "created_at": datetime.fromtimestamp(
            audio_path.stat().st_mtime,
            timezone.utc,
        ).isoformat(),
        "content_type": EXTENSION_AUDIO_TYPES[audio_path.suffix[1:]],
        "duration_seconds": None,
        "transcript": None,
        "language": None,
    }


def write_recording_metadata(recording_id: str, details: dict) -> None:
    temporary_path = RECORDINGS_DIR / f".{recording_id}.{uuid4().hex}.json.tmp"
    try:
        with temporary_path.open("w", encoding="utf-8") as details_file:
            json.dump(details, details_file)
        temporary_path.replace(metadata_path(recording_id))
    finally:
        temporary_path.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
    )


def transcribe_audio_file(audio_path: Path) -> tuple[str, str | None]:
    with TRANSCRIPTION_LOCK:
        model = get_whisper_model()
        segments, info = model.transcribe(
            str(audio_path),
            language=None,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, info.language


def recording_response(audio_path: Path) -> RecordingOut:
    recording_id = audio_path.stem
    details = read_recording_metadata(audio_path)
    created_at = datetime.fromisoformat(details["created_at"])
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    return RecordingOut(
        id=recording_id,
        created_at=created_at,
        content_type=details["content_type"],
        size_bytes=audio_path.stat().st_size,
        duration_seconds=details.get("duration_seconds"),
        transcript=details.get("transcript"),
        language=details.get("language"),
        audio_url=f"/api/recordings/{recording_id}/audio",
        download_url=f"/api/recordings/{recording_id}/download",
    )


def find_recording(recording_id: UUID) -> Path:
    for extension in set(ALLOWED_AUDIO_TYPES.values()):
        audio_path = RECORDINGS_DIR / f"{recording_id}.{extension}"
        if audio_path.is_file():
            return audio_path
    raise HTTPException(status_code=404, detail="Recording not found")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post(
    "/api/recordings",
    response_model=RecordingOut,
    status_code=201,
    summary="Save a recording locally",
)
def save_recording(
    file: Annotated[UploadFile, File(description="Recorded audio file")],
    duration_seconds: Annotated[float | None, Form(ge=0)] = None,
) -> RecordingOut:
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    extension = ALLOWED_AUDIO_TYPES.get(content_type)
    if extension is None:
        file.file.close()
        raise HTTPException(status_code=415, detail="Unsupported audio format")

    recording_id = str(uuid4())
    temporary_audio = RECORDINGS_DIR / f".{recording_id}.tmp"
    temporary_metadata = RECORDINGS_DIR / f".{recording_id}.json.tmp"
    audio_path = RECORDINGS_DIR / f"{recording_id}.{extension}"
    size_bytes = 0

    try:
        with temporary_audio.open("wb") as output:
            while chunk := file.file.read(CHUNK_SIZE):
                size_bytes += len(chunk)
                if size_bytes > MAX_RECORDING_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Recording exceeds the 100 MiB upload limit",
                    )
                output.write(chunk)

        if size_bytes == 0:
            raise HTTPException(status_code=400, detail="Recording is empty")

        details = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "content_type": content_type,
            "duration_seconds": duration_seconds,
        }
        with temporary_metadata.open("w", encoding="utf-8") as details_file:
            json.dump(details, details_file)

        temporary_audio.replace(audio_path)
        try:
            temporary_metadata.replace(metadata_path(recording_id))
        except OSError:
            audio_path.unlink(missing_ok=True)
            raise
    finally:
        temporary_audio.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        file.file.close()

    return recording_response(audio_path)


@app.post(
    "/api/recordings/{recording_id}/transcribe",
    response_model=TranscriptionOut,
    summary="Transcribe a saved recording",
)
def transcribe_recording(recording_id: UUID) -> TranscriptionOut:
    audio_path = find_recording(recording_id)
    transcript, language = transcribe_audio_file(audio_path)
    details = read_recording_metadata(audio_path)
    details["transcript"] = transcript
    details["language"] = language
    write_recording_metadata(str(recording_id), details)
    return TranscriptionOut(
        recording_id=str(recording_id),
        text=transcript,
        language=language,
    )


@app.get(
    "/api/recordings",
    response_model=list[RecordingOut],
    summary="List saved recordings",
)
def list_recordings() -> list[RecordingOut]:
    recordings = []
    for audio_path in RECORDINGS_DIR.iterdir():
        if audio_path.suffix[1:] not in EXTENSION_AUDIO_TYPES:
            continue
        try:
            UUID(audio_path.stem)
        except ValueError:
            continue
        recordings.append(recording_response(audio_path))

    return sorted(recordings, key=lambda recording: recording.created_at, reverse=True)


@app.get(
    "/api/recordings/{recording_id}/audio",
    summary="Play a saved recording",
)
def play_recording(recording_id: UUID) -> FileResponse:
    audio_path = find_recording(recording_id)
    recording = recording_response(audio_path)
    return FileResponse(audio_path, media_type=recording.content_type)


@app.get(
    "/api/recordings/{recording_id}/download",
    summary="Download a saved recording",
)
def download_recording(recording_id: UUID) -> FileResponse:
    audio_path = find_recording(recording_id)
    recording = recording_response(audio_path)
    return FileResponse(
        audio_path,
        media_type=recording.content_type,
        filename=f"recording-{recording_id}{audio_path.suffix}",
    )
