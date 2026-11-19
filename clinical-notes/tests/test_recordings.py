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

    def test_upload_list_play_and_download(self) -> None:
        audio = b"sample audio bytes"
        response = self.client.post(
            "/api/recordings",
            files={"file": ("sample.webm", audio, "audio/webm;codecs=opus")},
            data={"duration_seconds": "4.5"},
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
        self.assertEqual(self.client.get(recording["audio_url"]).content, audio)

        download = self.client.get(recording["download_url"])
        self.assertEqual(download.content, audio)
        self.assertIn("attachment", download.headers["content-disposition"])

        with patch(
            "main.transcribe_audio_file",
            return_value=("Hello from the recording.", "en"),
        ):
            transcription = self.client.post(
                f"/api/recordings/{recording['id']}/transcribe"
            )
        self.assertEqual(transcription.status_code, 200)
        self.assertEqual(transcription.json()["text"], "Hello from the recording.")
        self.assertEqual(transcription.json()["language"], "en")
        self.assertEqual(
            self.client.get("/api/recordings").json()[0]["transcript"],
            "Hello from the recording.",
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
