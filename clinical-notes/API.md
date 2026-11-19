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

### `GET /api/recordings`

List saved recordings, newest first.

### `GET /api/recordings/{id}/audio`

Play a saved recording.

### `GET /api/recordings/{id}/download`

Download a saved recording.

### `POST /api/recordings/{id}/transcribe`

Transcribe a saved recording with the local faster-whisper `small` model.
Language is detected automatically. The returned transcript and detected
language are saved in that recording's JSON sidecar.

Unsupported audio types return `415`; empty uploads return `400`; files larger
than 100 MiB return `413`; unknown recording IDs return `404`.

The model weights download the first time a recording is transcribed, so an
internet connection is required for that initial run. Transcription runs on
the local CPU; the audio is not sent to a transcription API.
