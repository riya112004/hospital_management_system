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
unless its transcript has already been reviewed. Pass `preserve_edits=true` to
skip the final pass when the user has unsaved transcript edits; the stored
transcript is returned as-is. Reviewed transcripts are returned as-is and are
not overwritten by later transcription passes. After
saving a recording, the browser runs this final pass and replaces the
provisional live transcript when the result is usable and the transcript
remains unreviewed. If the final result is empty or contains a repeated-phrase
loop, the existing live transcript is kept.
The browser can pass `language_mode=auto`, `language_mode=expected` with a
comma-separated `expected_languages` set, or `language_mode=single` with a
forced `language`. The expected Hindi-and-English combination also sends
`language=hi&prompt_style=hinglish`, which applies a fixed Roman-script example
and medical vocabulary to the final pass.

Unsupported audio types return `415`; empty uploads return `400`; files larger
than 100 MiB return `413`; unknown recording IDs return `404`.

The live `tiny` model is loaded and warmed when the server starts; a failed
warm-up is logged and retried on first use. The final
model weights download the first time a recording is transcribed, so an
internet connection is required for those initial runs. Transcription uses the
local CPU or an available GPU; the audio is not sent to a transcription API.
Final decoding disables previous-text conditioning and uses word timestamps
with a two-second hallucination-silence threshold. The default temperature
fallback remains enabled.

### `WebSocket /api/recordings/live?sample_rate=48000&language=en`

Stream mono PCM16 audio as binary WebSocket messages. The sample rate must be
between 8,000 and 96,000 Hz, and each audio message must be no longer than ten
seconds. `language_mode=auto` detects across languages,
`language_mode=expected` detects only within comma-separated
`expected_languages`, and `language_mode=single` requires a forced `language`.
If `language_mode` is omitted, the legacy automatic/optional-language behavior
is retained. In expected mode, the Hindi-and-English pair uses `language=hi`
and `prompt_style=hinglish`.
Each message is transcribed locally and returns JSON containing the cumulative
`text`, latest `segment`, any unstable-tail `unconfirmed` words, and detected
`language`. Unconfirmed words are shown separately in grey in the recorder and
are committed when a later window confirms them or the stream is stopped. Send a
`{"type":"resume","transcript":"...","resume_at_seconds":2,"language":"en"}`
text message after the ready status when reconnecting. This seeds the full
transcript, the audio position, and optional user-selected language for the
resumed stream. Transcript text is not fed back into the model prompt. Unknown
language codes in the query or resume state are ignored. Send
`{"type":"stop"}` as a text message to finish the stream. The
server commits pending unstable-tail words, sends one final transcript message,
and then closes the WebSocket.

The ready message provides `window_seconds`, `overlap_seconds`, and
`unstable_tail_seconds` from `config.py`; the browser uses the window and
overlap values to size its rolling audio windows. The server commits words
after the configured unstable tail and deduplicates words inside the overlap
by timestamp, preserving repeated words. Live transcription uses the `tiny`
model, one decoding beam, and word timestamps. VAD filtering is disabled for
the short live windows to reduce delay; very low-level windows are skipped
before inference. Automatic and expected-language modes detect on non-silent
windows, restrict expected-mode choices to the selected set, and require two
consecutive confident windows before switching languages. Hinglish mode uses
a fixed Roman-script example and medical vocabulary prompt.
The browser captures samples with an `AudioWorklet`, buffers them in a typed
ring buffer, and displays microphone level and live connection status. No more
than two windows are sent ahead; the browser waits for acknowledgements before
sending more and reports a lagging state instead of falling further behind.
Up to two live inference requests can run concurrently, independently of the
serialized final transcription model. Both models use the configured
`WHISPER_CPU_THREADS` budget, divided among concurrent workers; the final
transcription endpoint uses `small`. The final recording
is saved locally through `POST /api/recordings`.

The browser offers Auto, Expected languages (Hindi and English checked by
default), and Single language modes. A selected language mode applies to both
live and final transcription.
