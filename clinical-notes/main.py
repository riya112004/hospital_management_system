import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import tempfile
from threading import Lock
from typing import Annotated, AsyncIterator
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
from pydantic import BaseModel, Field

from config import (
    ALLOWED_AUDIO_TYPES,
    APP_DESCRIPTION,
    APP_TITLE,
    CHUNK_SIZE,
    LIVE_CHUNK_SECONDS,
    LIVE_MEDICAL_PROMPT,
    LIVE_OVERLAP_SECONDS,
    LIVE_PROMPT_CONTEXT_WORDS,
    LIVE_UNSTABLE_TAIL_SECONDS,
    LIVE_WHISPER_MODEL_SIZE,
    MAX_RECORDING_BYTES,
    MAX_TRANSCRIPT_LENGTH,
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
FINAL_TRANSCRIPTION_LOCK = Lock()
RECORDING_METADATA_LOCK = Lock()
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


def warm_live_model() -> None:
    get_live_whisper_model()
    silence = b"\x00\x00" * int(16_000 * LIVE_CHUNK_SECONDS)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                transcribe_pcm_chunk,
                silence,
                16_000,
                None,
                LIVE_MEDICAL_PROMPT,
            )
            for _ in range(2)
        ]
        for future in futures:
            future.result()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await asyncio.to_thread(warm_live_model)
    yield


app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RecordingOut(BaseModel):
    id: str
    created_at: datetime
    content_type: str
    size_bytes: int
    duration_seconds: float | None
    transcript: str | None
    transcript_reviewed: bool
    language: str | None
    audio_url: str
    download_url: str


class TranscriptionOut(BaseModel):
    recording_id: str
    text: str
    language: str | None
    transcript_reviewed: bool


class TranscriptUpdate(BaseModel):
    transcript: Annotated[str, Field(max_length=MAX_TRANSCRIPT_LENGTH)]


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
    return create_whisper_model(LIVE_WHISPER_MODEL_SIZE, num_workers=2)


def create_whisper_model(model_size: str, *, num_workers: int = 1) -> WhisperModel:
    return WhisperModel(
        model_size,
        device="auto",
        compute_type="int8",
        cpu_threads=max(1, WHISPER_CPU_THREADS // num_workers),
        num_workers=num_workers,
    )


def has_repeated_phrase_loop(text: str) -> bool:
    words = re.findall(r"\w+", text.casefold())
    if len(words) < 12:
        return False

    for phrase_size in range(min(12, len(words) // 2), 5, -1):
        phrase_counts = Counter(
            tuple(words[index : index + phrase_size])
            for index in range(len(words) - phrase_size + 1)
        )
        if any(
            count >= 2 and count * phrase_size >= len(words) * 0.4
            for count in phrase_counts.values()
        ):
            return True
    return False


def transcribe_audio_file(
    audio_path: Path,
    *,
    language: str | None = None,
    vad_filter: bool = True,
) -> tuple[str, str | None]:
    with FINAL_TRANSCRIPTION_LOCK:
        model = get_whisper_model()
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=vad_filter,
            condition_on_previous_text=False,
            word_timestamps=True,
            hallucination_silence_threshold=2,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, info.language


def transcribe_live_audio_file(
    audio_path: Path,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> tuple[str, str | None, list[tuple[float, float, str]]]:
    model = get_live_whisper_model()
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=False,
        word_timestamps=True,
        beam_size=1,
        initial_prompt=initial_prompt or LIVE_MEDICAL_PROMPT,
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
    initial_prompt: str | None = None,
) -> tuple[str, str | None, list[tuple[float, float, str]]]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_path = Path(audio_file.name)

    try:
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)
        return transcribe_live_audio_file(audio_path, language, initial_prompt)
    finally:
        audio_path.unlink(missing_ok=True)


def build_live_initial_prompt(recent_transcript: str) -> str:
    recent_words = recent_transcript.split()[-LIVE_PROMPT_CONTEXT_WORDS:]
    if not recent_words:
        return LIVE_MEDICAL_PROMPT
    return f"{LIVE_MEDICAL_PROMPT} Recent transcript: {' '.join(recent_words)}"


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
        transcript_reviewed=bool(details.get("transcript_reviewed", False)),
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
            "transcript_reviewed": False,
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
    requested_language = (
        language if language in TRANSCRIPTION_LANGUAGE_CODES else None
    )

    await websocket.accept()
    await asyncio.to_thread(get_live_whisper_model)
    try:
        await websocket.send_json(
            {
                "type": "status",
                "status": "ready",
                "window_seconds": LIVE_CHUNK_SECONDS,
                "overlap_seconds": LIVE_OVERLAP_SECONDS,
                "unstable_tail_seconds": LIVE_UNSTABLE_TAIL_SECONDS,
            }
        )
    except WebSocketDisconnect:
        return

    transcript = ""
    transcript_words: list[str] = []
    pending_words: list[tuple[float, float, str]] = []
    pending_word_tolerance = 0.08
    detected_language = requested_language
    chunk_start = 0.0
    last_word_end = 0.0
    resume_received = False

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
                final_words: list[str] = []
                for absolute_start, absolute_end, word_text in sorted(
                    pending_words,
                    key=lambda word: (word[0], word[1]),
                ):
                    if absolute_end <= last_word_end + pending_word_tolerance:
                        continue
                    if (
                        absolute_start < last_word_end - pending_word_tolerance
                        and transcript_words
                        and transcript_words[-1].casefold() == word_text.casefold()
                    ):
                        continue
                    final_words.append(word_text)
                    last_word_end = max(last_word_end, absolute_end)

                transcript_words.extend(final_words)
                transcript = " ".join(transcript_words)
                try:
                    await websocket.send_json(
                        {
                            "type": "transcript",
                            "text": transcript,
                            "segment": " ".join(final_words),
                            "language": detected_language,
                        }
                    )
                except WebSocketDisconnect:
                    return
                await websocket.close()
                return
            if command.get("type") == "resume" and not resume_received:
                resume_text = command.get("transcript", "")
                resume_at = command.get("resume_at_seconds", 0)
                resume_language = command.get("language")
                if (
                    not isinstance(resume_text, str)
                    or isinstance(resume_at, bool)
                    or not isinstance(resume_at, (int, float))
                    or resume_at < 0
                    or resume_at > 1_000_000_000
                    or not math.isfinite(resume_at)
                    or (
                        resume_language is not None
                        and not isinstance(resume_language, str)
                    )
                ):
                    await websocket.close(
                        code=1008,
                        reason="Invalid live transcription resume state",
                    )
                    return
                transcript_words = resume_text.split()[-LIVE_PROMPT_CONTEXT_WORDS:]
                transcript = " ".join(transcript_words)
                pending_words = []
                chunk_start = float(resume_at)
                last_word_end = (
                    max(
                        0.0,
                        chunk_start
                        + LIVE_OVERLAP_SECONDS
                        - LIVE_UNSTABLE_TAIL_SECONDS,
                    )
                    if chunk_start > 0
                    else 0.0
                )
                if resume_language in TRANSCRIPTION_LANGUAGE_CODES:
                    requested_language = resume_language
                    detected_language = resume_language
                resume_received = True
                continue
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
        pending_word_tolerance = tolerance

        if not any(pcm):
            try:
                await websocket.send_json(
                    {
                        "type": "transcript",
                        "text": transcript,
                        "segment": "",
                        "language": detected_language,
                    }
                )
            except WebSocketDisconnect:
                return
            continue

        segment, chunk_language, words = await asyncio.to_thread(
            transcribe_pcm_chunk,
            pcm,
            sample_rate,
            requested_language,
            build_live_initial_prompt(transcript),
        )
        if chunk_language:
            detected_language = chunk_language

        if words:
            new_words: list[str] = []
            window_pending_words: list[tuple[float, float, str]] = []
            for word_start, word_end, word_text in words:
                absolute_start = current_start + word_start
                absolute_end = current_start + word_end
                if word_end > stable_until:
                    if word_end <= chunk_seconds:
                        window_pending_words.append(
                            (absolute_start, absolute_end, word_text)
                        )
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
            pending_words = [
                pending_word
                for pending_word in pending_words
                if pending_word[1] > last_word_end + tolerance
            ]
            pending_words.extend(window_pending_words)
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
    with RECORDING_METADATA_LOCK:
        if not audio_path.is_file():
            raise HTTPException(status_code=404, detail="Recording not found")
        details = read_recording_metadata(audio_path)
        if details.get("transcript_reviewed"):
            return TranscriptionOut(
                recording_id=str(recording_id),
                text=details.get("transcript") or "",
                language=details.get("language"),
                transcript_reviewed=True,
            )
        live_transcript = details.get("transcript") or ""
        live_language = details.get("language")

    transcript, detected_language = transcribe_audio_file(
        audio_path,
        language=language,
    )
    if live_transcript and (
        not transcript.strip() or has_repeated_phrase_loop(transcript)
    ):
        transcript = live_transcript
        detected_language = live_language or detected_language

    with RECORDING_METADATA_LOCK:
        if not audio_path.is_file():
            raise HTTPException(status_code=404, detail="Recording not found")
        details = read_recording_metadata(audio_path)
        if details.get("transcript_reviewed"):
            transcript = details.get("transcript") or ""
            detected_language = details.get("language")
        else:
            details["transcript"] = transcript
            details["language"] = detected_language
            write_recording_metadata(str(recording_id), details)
    return TranscriptionOut(
        recording_id=str(recording_id),
        text=transcript,
        language=detected_language,
        transcript_reviewed=bool(details.get("transcript_reviewed", False)),
    )


@app.put(
    "/api/recordings/{recording_id}/transcript",
    response_model=RecordingOut,
    summary="Save edits to a recording transcript",
)
def update_recording_transcript(
    recording_id: UUID,
    update: TranscriptUpdate,
) -> RecordingOut:
    with RECORDING_METADATA_LOCK:
        audio_path = find_recording(recording_id)
        details = read_recording_metadata(audio_path)
        details["transcript"] = update.transcript
        details["transcript_reviewed"] = True
        write_recording_metadata(str(recording_id), details)
        return recording_response(audio_path)


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


@app.delete(
    "/api/recordings/{recording_id}",
    status_code=204,
    summary="Delete a saved recording",
)
def delete_recording(recording_id: UUID) -> None:
    with RECORDING_METADATA_LOCK:
        audio_path = find_recording(recording_id)
        try:
            audio_path.unlink()
            metadata_path(str(recording_id)).unlink(missing_ok=True)
        except OSError as error:
            raise HTTPException(
                status_code=500,
                detail="Could not delete the recording.",
            ) from error


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
