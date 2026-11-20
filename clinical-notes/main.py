import asyncio
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import tempfile
from threading import Lock
from typing import Annotated
from uuid import UUID, uuid4
import wave

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydantic import BaseModel

from config import (
    ALLOWED_AUDIO_TYPES,
    APP_DESCRIPTION,
    APP_TITLE,
    CHUNK_SIZE,
    LIVE_CHUNK_SECONDS,
    LIVE_OVERLAP_SECONDS,
    LIVE_UNSTABLE_TAIL_SECONDS,
    LIVE_WHISPER_MODEL_SIZE,
    MAX_RECORDING_BYTES,
    RECORDINGS_DIR,
    STATIC_DIR,
    TRANSCRIPTION_LANGUAGE_CODES,
    WHISPER_CPU_THREADS,
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
    return create_whisper_model(WHISPER_MODEL_SIZE)


@lru_cache(maxsize=1)
def get_live_whisper_model() -> WhisperModel:
    return create_whisper_model(LIVE_WHISPER_MODEL_SIZE)


def create_whisper_model(model_size: str) -> WhisperModel:
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=WHISPER_CPU_THREADS,
    )


def transcribe_audio_file(
    audio_path: Path,
    *,
    language: str | None = None,
    vad_filter: bool = True,
) -> tuple[str, str | None]:
    with TRANSCRIPTION_LOCK:
        model = get_whisper_model()
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=vad_filter,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, info.language


def transcribe_live_audio_file(
    audio_path: Path,
    language: str | None = None,
) -> tuple[str, str | None, list[tuple[float, float, str]]]:
    with TRANSCRIPTION_LOCK:
        model = get_live_whisper_model()
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=False,
            word_timestamps=True,
            beam_size=1,
        )
        words: list[tuple[float, float, str]] = []
        for segment in segments:
            for word in getattr(segment, "words", None) or []:
                word_text = (getattr(word, "word", "") or "").strip()
                if word_text:
                    words.append(
                        (float(word.start), float(word.end), word_text),
                    )

    text = " ".join(word for _, _, word in words)
    return text, info.language, words


def transcribe_pcm_chunk(
    pcm: bytes,
    sample_rate: int,
    language: str | None = None,
) -> tuple[str, str | None, list[tuple[float, float, str]]]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_path = Path(audio_file.name)

    try:
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)
        return transcribe_live_audio_file(audio_path, language)
    finally:
        audio_path.unlink(missing_ok=True)


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
    transcript: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
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
            "transcript": transcript,
            "language": language,
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


@app.websocket("/api/recordings/live")
async def transcribe_live(
    websocket: WebSocket,
    sample_rate: int = 48000,
    language: str | None = None,
) -> None:
    if not 8000 <= sample_rate <= 96000:
        await websocket.close(code=1008, reason="Unsupported audio sample rate")
        return
    if language is not None and language not in TRANSCRIPTION_LANGUAGE_CODES:
        await websocket.close(code=1008, reason="Unsupported transcription language")
        return
    requested_language = language or None

    await websocket.accept()
    await asyncio.to_thread(get_live_whisper_model)
    try:
        await websocket.send_json(
            {
                "type": "status",
                "status": "ready",
                "window_seconds": LIVE_CHUNK_SECONDS,
                "overlap_seconds": LIVE_OVERLAP_SECONDS,
            }
        )
    except WebSocketDisconnect:
        return

    transcript = ""
    transcript_words: list[str] = []
    detected_language = requested_language
    chunk_start = 0.0
    last_word_end = 0.0

    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return

        if message.get("text") is not None:
            try:
                command = json.loads(message["text"])
            except json.JSONDecodeError:
                await websocket.close(code=1003, reason="Expected a JSON command")
                return
            if not isinstance(command, dict):
                await websocket.close(code=1003, reason="Expected a JSON object")
                return
            if command.get("type") == "stop":
                await websocket.close()
                return
            await websocket.close(code=1003, reason="Unsupported command")
            return

        pcm = message.get("bytes")
        if not pcm or len(pcm) % 2:
            await websocket.close(code=1003, reason="Expected non-empty PCM audio")
            return
        if len(pcm) > sample_rate * 2 * 10:
            await websocket.close(code=1009, reason="Audio chunk exceeds 10 seconds")
            return

        chunk_seconds = len(pcm) / (2 * sample_rate)
        current_start = chunk_start
        chunk_start += LIVE_CHUNK_SECONDS - LIVE_OVERLAP_SECONDS
        stable_until = chunk_seconds - LIVE_UNSTABLE_TAIL_SECONDS
        tolerance = min(0.08, chunk_seconds * 0.05)

        if not any(pcm):
            continue

        segment, chunk_language, words = await asyncio.to_thread(
            transcribe_pcm_chunk,
            pcm,
            sample_rate,
            requested_language,
        )
        if chunk_language:
            detected_language = chunk_language

        if words:
            new_words: list[str] = []
            for word_start, word_end, word_text in words:
                absolute_start = current_start + word_start
                absolute_end = current_start + word_end
                if word_end > stable_until:
                    continue
                if absolute_end <= last_word_end + tolerance:
                    continue
                if (
                    absolute_start < last_word_end - tolerance
                    and transcript_words
                    and transcript_words[-1].casefold() == word_text.casefold()
                ):
                    continue
                new_words.append(word_text)
                last_word_end = max(last_word_end, absolute_end)
            transcript_words.extend(new_words)
            transcript = " ".join(transcript_words)
            segment = " ".join(new_words)
        else:
            segment = ""

        try:
            await websocket.send_json(
                {
                    "type": "transcript",
                    "text": transcript,
                    "segment": segment,
                    "language": detected_language,
                }
            )
        except WebSocketDisconnect:
            return


@app.post(
    "/api/recordings/{recording_id}/transcribe",
    response_model=TranscriptionOut,
    summary="Transcribe a saved recording",
)
def transcribe_recording(
    recording_id: UUID,
    language: str | None = None,
) -> TranscriptionOut:
    if language is not None and language not in TRANSCRIPTION_LANGUAGE_CODES:
        raise HTTPException(status_code=422, detail="Unsupported transcription language")
    audio_path = find_recording(recording_id)
    transcript, detected_language = transcribe_audio_file(
        audio_path,
        language=language,
    )
    details = read_recording_metadata(audio_path)
    details["transcript"] = transcript
    details["language"] = detected_language
    write_recording_metadata(str(recording_id), details)
    return TranscriptionOut(
        recording_id=str(recording_id),
        text=transcript,
        language=detected_language,
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
