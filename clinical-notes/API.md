# Recording API

Recordings are saved on this computer under `data/recordings/`. Each recording
has an audio file and a small JSON sidecar containing its timestamp and
playback metadata. No database connection is used.

## Endpoints

### `POST /api/recordings`

Save an audio recording as `multipart/form-data`.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `file` | File | Yes | Audio file (`webm`, `ogg`, `mp4`, `mpeg`, `wav`, `aac`, or `flac`) |
| `duration_seconds` | Number | No | Recording duration in seconds |
| `transcript` | Text | No | Live transcript to save with the recording |
| `language` | Text | No | Detected language for the live transcript |

### `GET /api/recordings`

List saved recordings, newest first.

### `GET /api/recordings/{id}/audio`

Play a saved recording.

### `GET /api/recordings/{id}/download`

Download a saved recording.

### `POST /api/recordings/{id}/transcribe`

Transcribe a saved recording with the local faster-whisper `large-v3-turbo` model.
Pass an optional `language` query parameter (for example, `?language=pt`) to
force a language; otherwise it is detected automatically. The returned
transcript and detected language are saved in that recording's JSON sidecar.
After saving a recording, the browser runs this final pass and replaces the
provisional live transcript when transcription succeeds.

Unsupported audio types return `415`; empty uploads return `400`; files larger
than 100 MiB return `413`; unknown recording IDs return `404`.

The model weights download the first time a recording is transcribed, so an
internet connection is required for that initial run. Transcription runs on
the local CPU; the audio is not sent to a transcription API.

### `WebSocket /api/recordings/live?sample_rate=48000&language=en`

Stream mono PCM16 audio as binary WebSocket messages. The sample rate must be
between 8,000 and 96,000 Hz, and each audio message must be no longer than ten
seconds. The optional `language` code forces transcription to that language;
if omitted, it is detected automatically. Each message is transcribed locally
and returns JSON containing the cumulative `text`, latest `segment`, and
detected `language`. Send `{"type":"stop"}` as a text message to finish the
stream.

The browser recorder sends rolling four-second windows with two seconds of
overlap (a new window every two seconds). Live transcription uses the `base`
model, one decoding beam, and word timestamps. Words within the final 0.5
seconds of each window are held back until a later window confirms them; words
inside the overlap are deduplicated by timestamp, preserving repeated words.
The window, overlap, and unstable-tail lengths are controlled by
`LIVE_CHUNK_SECONDS`, `LIVE_OVERLAP_SECONDS`, and
`LIVE_UNSTABLE_TAIL_SECONDS` in `config.py`. The live and final models share
the configured `WHISPER_CPU_THREADS`; the final transcription endpoint
continues to use `large-v3-turbo`. The final recording is saved locally through
`POST /api/recordings`.

The browser offers automatic detection plus English, Portuguese, Hindi,
Spanish, French, German, Italian, Arabic, Chinese, and Japanese. A selected
language is used for both live and final transcription.
