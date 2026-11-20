import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import wave

from faster_whisper.audio import decode_audio
from fastapi.testclient import TestClient

import main


class RecordingApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_recordings_dir = main.RECORDINGS_DIR
        cls.temp_dir = tempfile.TemporaryDirectory()
        main.RECORDINGS_DIR = Path(cls.temp_dir.name)
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        main.RECORDINGS_DIR = cls.original_recordings_dir
        cls.temp_dir.cleanup()

    def test_live_and_final_models_use_configured_sizes_and_cpu_threads(self) -> None:
        main.get_whisper_model.cache_clear()
        main.get_live_whisper_model.cache_clear()
        try:
            with patch("main.WhisperModel") as model_factory:
                main.get_whisper_model()
                main.get_live_whisper_model()
        finally:
            main.get_whisper_model.cache_clear()
            main.get_live_whisper_model.cache_clear()

        self.assertEqual(
            [call.args[0] for call in model_factory.call_args_list],
            [main.WHISPER_MODEL_SIZE, main.LIVE_WHISPER_MODEL_SIZE],
        )
        self.assertTrue(
            all(
                call.kwargs["cpu_threads"] == main.WHISPER_CPU_THREADS
                for call in model_factory.call_args_list
            )
        )

    def test_upload_list_play_and_download(self) -> None:
        audio = b"sample audio bytes"
        response = self.client.post(
            "/api/recordings",
            files={"file": ("sample.webm", audio, "audio/webm;codecs=opus")},
            data={
                "duration_seconds": "4.5",
                "transcript": "Live words",
                "language": "en",
            },
        )

        self.assertEqual(response.status_code, 201)
        recording = response.json()
        self.assertEqual(recording["duration_seconds"], 4.5)
        self.assertEqual(recording["size_bytes"], len(audio))

        saved_path = main.RECORDINGS_DIR / f"{recording['id']}.webm"
        self.assertEqual(saved_path.read_bytes(), audio)
        self.assertTrue((main.RECORDINGS_DIR / f"{recording['id']}.json").is_file())

        listing = self.client.get("/api/recordings")
        self.assertEqual(listing.status_code, 200)
        self.assertIn(recording["id"], [item["id"] for item in listing.json()])
        self.assertEqual(listing.json()[0]["transcript"], "Live words")
        self.assertEqual(self.client.get(recording["audio_url"]).content, audio)

        download = self.client.get(recording["download_url"])
        self.assertEqual(download.content, audio)
        self.assertIn("attachment", download.headers["content-disposition"])

        with patch(
            "main.transcribe_audio_file",
            return_value=("Hello from the recording.", "en"),
        ) as transcribe_file:
            transcription = self.client.post(
                f"/api/recordings/{recording['id']}/transcribe?language=pt"
            )
        transcribe_file.assert_called_once_with(
            saved_path,
            language="pt",
        )
        self.assertEqual(transcription.status_code, 200)
        self.assertEqual(transcription.json()["text"], "Hello from the recording.")
        self.assertEqual(transcription.json()["language"], "en")
        self.assertEqual(
            self.client.get("/api/recordings").json()[0]["transcript"],
            "Hello from the recording.",
        )

    def test_live_transcription_commits_stable_words_from_rolling_windows(self) -> None:
        sample_rate = 8000
        window = b"\x01\x00" * int(sample_rate * main.LIVE_CHUNK_SECONDS)
        with (
            patch("main.get_live_whisper_model"),
            patch(
                "main.transcribe_pcm_chunk",
                side_effect=[
                    (
                        "Hello there Watch this.",
                        "en",
                        [
                            (0.1, 0.2, "Hello"),
                            (2.1, 2.2, "there"),
                            (3.2, 3.4, "Watch"),
                            (3.55, 3.9, "this."),
                        ],
                    ),
                    (
                        "there this. friend.",
                        "en",
                        [
                            (0.1, 0.2, "there"),
                            (1.55, 1.9, "this."),
                            (2.2, 2.4, "friend."),
                        ],
                    ),
                ],
            ),
        ):
            with self.client.websocket_connect(
                f"/api/recordings/live?sample_rate={sample_rate}"
            ) as websocket:
                ready = websocket.receive_json()
                self.assertEqual(ready["type"], "status")
                self.assertEqual(ready["window_seconds"], main.LIVE_CHUNK_SECONDS)
                self.assertEqual(ready["overlap_seconds"], main.LIVE_OVERLAP_SECONDS)
                websocket.send_bytes(window)
                first = websocket.receive_json()
                websocket.send_bytes(window)
                second = websocket.receive_json()
                websocket.send_text('{"type":"stop"}')

        self.assertEqual(first["text"], "Hello there Watch")
        self.assertEqual(second["text"], "Hello there Watch this. friend.")
        self.assertEqual(second["language"], "en")
        self.assertEqual(second["segment"], "this. friend.")

    def test_live_transcription_uses_word_timestamps(self) -> None:
        class FakeModel:
            def transcribe(
                self,
                audio_path,
                *,
                language,
                vad_filter,
                word_timestamps,
                beam_size,
            ):
                self.language = language
                self.vad_filter = vad_filter
                self.word_timestamps = word_timestamps
                self.beam_size = beam_size
                words = [
                    SimpleNamespace(word=" Hello", start=0.0, end=0.2),
                    SimpleNamespace(word=" there", start=0.3, end=0.5),
                ]
                segment = SimpleNamespace(text=" Hello there", words=words)
                return iter([segment]), SimpleNamespace(language="pt")

        model = FakeModel()
        with patch("main.get_live_whisper_model", return_value=model):
            result = main.transcribe_live_audio_file(
                Path(self.temp_dir.name) / "sample.webm",
                "pt",
            )

        self.assertEqual(result[0], "Hello there")
        self.assertEqual(result[1], "pt")
        self.assertEqual(result[2], [(0.0, 0.2, "Hello"), (0.3, 0.5, "there")])
        self.assertFalse(model.vad_filter)
        self.assertTrue(model.word_timestamps)
        self.assertEqual(model.beam_size, 1)
        self.assertEqual(model.language, "pt")

    def test_live_transcription_keeps_repeated_words(self) -> None:
        sample_rate = 8000
        window = b"\x01\x00" * int(sample_rate * main.LIVE_CHUNK_SECONDS)
        with (
            patch("main.get_live_whisper_model"),
            patch(
                "main.transcribe_pcm_chunk",
                side_effect=[
                    (
                        "no no",
                        "en",
                        [(0.1, 0.2, "no"), (2.1, 2.2, "no")],
                    ),
                    (
                        "no yes",
                        "en",
                        [(0.1, 0.2, "no"), (0.5, 0.6, "yes")],
                    ),
                ],
            ),
        ):
            with self.client.websocket_connect(
                f"/api/recordings/live?sample_rate={sample_rate}"
            ) as websocket:
                websocket.receive_json()
                websocket.send_bytes(window)
                websocket.receive_json()
                websocket.send_bytes(window)
                second = websocket.receive_json()
                websocket.send_text('{"type":"stop"}')

        self.assertEqual(second["text"], "no no yes")

    def test_live_pcm_chunk_is_wrapped_as_wav(self) -> None:
        def verify_wav(
            audio_path: Path,
            language: str | None,
        ) -> tuple[str, str, list[tuple[float, float, str]]]:
            self.assertIsNone(language)
            with wave.open(str(audio_path), "rb") as audio_file:
                self.assertEqual(audio_file.getnchannels(), 1)
                self.assertEqual(audio_file.getsampwidth(), 2)
                self.assertEqual(audio_file.getframerate(), 16000)
                self.assertEqual(audio_file.readframes(2), b"\x01\x00" * 2)
            return "Hello", "en", [(0.0, 0.1, "Hello")]

        with patch("main.transcribe_live_audio_file", side_effect=verify_wav):
            result = main.transcribe_pcm_chunk(b"\x01\x00" * 2, 16000)

        self.assertEqual(result, ("Hello", "en", [(0.0, 0.1, "Hello")]))

    def test_live_websocket_passes_selected_language_to_transcription(self) -> None:
        sample_rate = 8000
        window = b"\x01\x00" * int(sample_rate * main.LIVE_CHUNK_SECONDS)
        with (
            patch("main.get_live_whisper_model"),
            patch(
                "main.transcribe_pcm_chunk",
                return_value=("Olá", "pt", [(0.1, 0.3, "Olá")]),
            ) as transcribe_chunk,
        ):
            with self.client.websocket_connect(
                f"/api/recordings/live?sample_rate={sample_rate}&language=pt"
            ) as websocket:
                websocket.receive_json()
                websocket.send_bytes(window)
                self.assertEqual(websocket.receive_json()["language"], "pt")
                websocket.send_text('{"type":"stop"}')

        transcribe_chunk.assert_called_once()
        self.assertEqual(
            transcribe_chunk.call_args.args[1:],
            (sample_rate, "pt"),
        )

    def test_invalid_uploads_and_unknown_recording(self) -> None:
        unsupported = self.client.post(
            "/api/recordings",
            files={"file": ("notes.txt", b"not audio", "text/plain")},
        )
        self.assertEqual(unsupported.status_code, 415)

        empty = self.client.post(
            "/api/recordings",
            files={"file": ("empty.webm", b"", "audio/webm")},
        )
        self.assertEqual(empty.status_code, 400)

        unknown = self.client.get(
            "/api/recordings/00000000-0000-0000-0000-000000000000/audio"
        )
        self.assertEqual(unknown.status_code, 404)

        invalid_language = self.client.post(
            "/api/recordings/00000000-0000-0000-0000-000000000000/transcribe"
            "?language=xx"
        )
        self.assertEqual(invalid_language.status_code, 422)

    def test_transcription_joins_segments_and_detects_language(self) -> None:
        class FakeModel:
            def transcribe(self, audio_path, *, language, vad_filter):
                self.audio_path = audio_path
                self.language = language
                self.vad_filter = vad_filter
                segments = [
                    SimpleNamespace(text=" Hello"),
                    SimpleNamespace(text=" world."),
                ]
                info = SimpleNamespace(language="en")
                return iter(segments), info

        model = FakeModel()
        with patch("main.get_whisper_model", return_value=model):
            transcript, language = main.transcribe_audio_file(
                Path(self.temp_dir.name) / "sample.webm"
            )

        self.assertEqual(transcript, "Hello world.")
        self.assertEqual(language, "en")
        self.assertEqual(model.language, None)
        self.assertTrue(model.vad_filter)

    def test_faster_whisper_decoder_accepts_audio(self) -> None:
        audio_path = Path(self.temp_dir.name) / "decoder-smoke.wav"
        with wave.open(str(audio_path), "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(16000)
            audio_file.writeframes(b"\x00\x00" * 1600)

        decoded = decode_audio(str(audio_path))
        self.assertGreater(decoded.size, 0)


if __name__ == "__main__":
    unittest.main()
