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

List saved recordings, newest first. Entries include duration, detected
language, and whether the transcript has been reviewed.

### `PUT /api/recordings/{id}/transcript`

Save reviewed transcript edits as JSON, for example
`{"transcript":"Reviewed clinical note."}`. An empty transcript is allowed.
Manual edits are limited to 50,000 characters. Saving edits marks the
transcript reviewed so later transcription passes do not replace it. Live and
final transcripts are not truncated by this edit limit.

### `DELETE /api/recordings/{id}`

Delete the saved audio file and its metadata. This cannot be undone.

### `GET /api/recordings/{id}/audio`

Play a saved recording.

### `GET /api/recordings/{id}/download`

Download a saved recording.

### `POST /api/recordings/{id}/transcribe`

Transcribe a saved recording with the local faster-whisper `small` model.
Pass an optional `language` query parameter (for example, `?language=pt`) to
force a language; otherwise it is detected automatically. The returned
transcript and detected language are saved in that recording's JSON sidecar
unless its transcript has already been reviewed. Reviewed transcripts are
returned as-is and are not overwritten by later transcription passes. After
saving a recording, the browser runs this final pass and replaces the
provisional live transcript when the result is usable and the transcript
remains unreviewed. If the final result is empty or contains a repeated-phrase
loop, the existing live transcript is kept.

Unsupported audio types return `415`; empty uploads return `400`; files larger
than 100 MiB return `413`; unknown recording IDs return `404`.

The live `tiny` model is loaded and warmed when the server starts. The final
model weights download the first time a recording is transcribed, so an
internet connection is required for those initial runs. Transcription uses the
local CPU or an available GPU; the audio is not sent to a transcription API.
Final decoding disables previous-text conditioning and uses word timestamps
with a two-second hallucination-silence threshold. The default temperature
fallback remains enabled.

### `WebSocket /api/recordings/live?sample_rate=48000&language=en`

Stream mono PCM16 audio as binary WebSocket messages. The sample rate must be
between 8,000 and 96,000 Hz, and each audio message must be no longer than ten
seconds. The optional `language` code forces transcription to that language;
if omitted or unsupported, each window detects its language independently.
Each message is transcribed locally and returns JSON containing the cumulative
`text`, latest `segment`, and detected `language`. Send a
`{"type":"resume","transcript":"...","resume_at_seconds":2,"language":"en"}`
text message after the ready status when reconnecting. This seeds recent
transcript context, the audio position, and optional user-selected language
for the resumed stream. Unknown language codes in the query or resume state
are ignored. Send `{"type":"stop"}` as a text message to finish the stream. The
server commits pending unstable-tail words, sends one final transcript message,
and then closes the WebSocket.

The ready message provides `window_seconds`, `overlap_seconds`, and
`unstable_tail_seconds` from `config.py`; the browser uses the window and
overlap values to size its rolling audio windows. The server commits words
after the configured unstable tail and deduplicates words inside the overlap
by timestamp, preserving repeated words. Live transcription uses the `tiny`
model, one decoding beam, and word timestamps. VAD filtering is disabled for
the short live windows to reduce delay. Each live window uses the recent
committed transcript and clinical-note vocabulary as its `initial_prompt`.
The browser captures samples with an `AudioWorklet`, buffers them in a typed
ring buffer, and displays microphone level and live connection status.
Up to two live inference requests can run concurrently, independently of the
serialized final transcription model. Both models use the configured
`WHISPER_CPU_THREADS` budget, divided among concurrent workers; the final
transcription endpoint uses `small`. The final recording
is saved locally through `POST /api/recordings`.

The browser offers automatic detection plus English, Portuguese, Hindi,
Spanish, French, German, Italian, Arabic, Chinese, and Japanese. A selected
language is used for both live and final transcription.
