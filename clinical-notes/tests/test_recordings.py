import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
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

    def test_live_and_final_models_use_auto_device_and_worker_counts(self) -> None:
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
        self.assertEqual(
            [call.kwargs["cpu_threads"] for call in model_factory.call_args_list],
            [
                main.WHISPER_CPU_THREADS,
                main.WHISPER_CPU_THREADS,
            ],
        )
        self.assertEqual(
            [call.kwargs["device"] for call in model_factory.call_args_list],
            ["auto", "auto"],
        )
        self.assertEqual(
            [call.kwargs["num_workers"] for call in model_factory.call_args_list],
            [1, 1],
        )

    def test_live_transcription_requests_can_run_concurrently(self) -> None:
        barrier = Barrier(2)

        class ConcurrentModel:
            def transcribe(self, *args, **kwargs):
                barrier.wait(timeout=5)
                return iter(()), SimpleNamespace(language="en")

        with patch("main.get_live_whisper_model", return_value=ConcurrentModel()):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(
                    executor.map(
                        lambda _: main.transcribe_live_audio_file(
                            Path(self.temp_dir.name) / "sample.wav"
                        ),
                        range(2),
                    )
                )

        self.assertEqual(results, [("", "en", []), ("", "en", [])])

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

        with patch("main.transcribe_audio_file") as transcribe_file:
            preserved = self.client.post(
                f"/api/recordings/{recording['id']}/transcribe?preserve_edits=true"
            )
        transcribe_file.assert_not_called()
        self.assertEqual(preserved.json()["text"], "Hello from the recording.")

        def finish_after_review(audio_path, *, language):
            details = main.read_recording_metadata(audio_path)
            details["transcript"] = "Reviewed clinical note."
            details["transcript_reviewed"] = True
            main.write_recording_metadata(recording["id"], details)
            return "Generated transcript.", "en"

        with patch(
            "main.transcribe_audio_file",
            side_effect=finish_after_review,
        ) as transcribe_file:
            reviewed_result = self.client.post(
                f"/api/recordings/{recording['id']}/transcribe"
            )
        transcribe_file.assert_called_once()
        self.assertEqual(
            reviewed_result.json()["text"],
            "Reviewed clinical note.",
        )
        self.assertTrue(reviewed_result.json()["transcript_reviewed"])

        edited = self.client.put(
            f"/api/recordings/{recording['id']}/transcript",
            json={"transcript": "Edited and reviewed note."},
        )
        self.assertEqual(edited.status_code, 200)
        self.assertEqual(edited.json()["transcript"], "Edited and reviewed note.")
        self.assertTrue(edited.json()["transcript_reviewed"])

        with patch("main.transcribe_audio_file") as transcribe_file:
            reviewed_result = self.client.post(
                f"/api/recordings/{recording['id']}/transcribe"
            )
        transcribe_file.assert_not_called()
        self.assertEqual(
            reviewed_result.json()["text"],
            "Edited and reviewed note.",
        )
        self.assertEqual(
            self.client.get("/api/recordings").json()[0]["transcript"],
            "Edited and reviewed note.",
        )

        longer_edit = "x" * 734
        longer_edit_result = self.client.put(
            f"/api/recordings/{recording['id']}/transcript",
            json={"transcript": longer_edit},
        )
        self.assertEqual(longer_edit_result.status_code, 200)
        self.assertEqual(longer_edit_result.json()["transcript"], longer_edit)

        maximum_length = "x" * main.MAX_TRANSCRIPT_LENGTH
        within_limit = self.client.put(
            f"/api/recordings/{recording['id']}/transcript",
            json={"transcript": maximum_length},
        )
        self.assertEqual(within_limit.status_code, 200)
        self.assertEqual(
            len(within_limit.json()["transcript"]),
            main.MAX_TRANSCRIPT_LENGTH,
        )

        over_limit = self.client.put(
            f"/api/recordings/{recording['id']}/transcript",
            json={"transcript": maximum_length + "x"},
        )
        self.assertEqual(over_limit.status_code, 422)
        self.assertEqual(
            self.client.get("/api/recordings").json()[0]["transcript"],
            maximum_length,
        )

        deleted = self.client.delete(f"/api/recordings/{recording['id']}")
        self.assertEqual(deleted.status_code, 204)
        self.assertFalse(saved_path.exists())
        self.assertFalse(
            (main.RECORDINGS_DIR / f"{recording['id']}.json").exists()
        )
        self.assertEqual(
            self.client.delete(f"/api/recordings/{recording['id']}").status_code,
            404,
        )

    def test_final_transcription_keeps_live_text_when_output_loops(self) -> None:
        live_transcript = (
            "The patient reports intermittent chest discomfort without "
            "shortness of breath."
        )
        response = self.client.post(
            "/api/recordings",
            files={"file": ("sample.webm", b"sample audio bytes", "audio/webm")},
            data={"transcript": live_transcript, "language": "en"},
        )
        self.assertEqual(response.status_code, 201)
        recording_id = response.json()["id"]
        repeated_transcript = (
            "The title of the story is the title of her. " * 4
        ).strip()

        with patch(
            "main.transcribe_audio_file",
            return_value=(repeated_transcript, "en"),
        ):
            result = self.client.post(
                f"/api/recordings/{recording_id}/transcribe"
            )

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["text"], live_transcript)
        self.assertEqual(
            self.client.get("/api/recordings").json()[0]["transcript"],
            live_transcript,
        )

    def test_hinglish_final_transcription_uses_hindi_and_fixed_prompt(self) -> None:
        recording = self.client.post(
            "/api/recordings",
            files={"file": ("sample.webm", b"sample audio bytes", "audio/webm")},
        ).json()
        audio_path = main.RECORDINGS_DIR / f"{recording['id']}.webm"

        with patch(
            "main.transcribe_audio_file",
            return_value=("Doctor, mujhe bukhar hai.", "hi"),
        ) as transcribe_file:
            response = self.client.post(
                f"/api/recordings/{recording['id']}/transcribe"
                "?language_mode=expected&expected_languages=hi,en"
                "&language=hi&prompt_style=hinglish"
            )

        self.assertEqual(response.status_code, 200)
        transcribe_file.assert_called_once_with(
            audio_path,
            language="hi",
            initial_prompt=main.LIVE_HINGLISH_PROMPT,
        )

    def test_expected_final_language_is_selected_only_from_allowed_codes(self) -> None:
        class FakeModel:
            def detect_language(self, *, audio):
                return "ta", 0.8, [
                    ("ta", 0.8),
                    ("pt", 0.12),
                    ("fr", 0.08),
                ]

            def transcribe(self, audio_path, **kwargs):
                self.kwargs = kwargs
                return iter(()), SimpleNamespace(language=kwargs["language"])

        model = FakeModel()
        with (
            patch("main.get_whisper_model", return_value=model),
            patch("main.decode_audio", return_value=object()),
        ):
            _, language = main.transcribe_audio_file(
                Path(self.temp_dir.name) / "sample.wav",
                allowed_languages=frozenset({"pt", "fr"}),
            )

        self.assertEqual(language, "fr")
        self.assertEqual(model.kwargs["language"], "fr")

    def test_final_transcription_preserves_transcripts_over_edit_limit(self) -> None:
        audio = self.client.post(
            "/api/recordings",
            files={"file": ("sample.webm", b"sample audio bytes", "audio/webm")},
        ).json()
        complete_transcript = "doctor spoke " * 70

        with patch(
            "main.transcribe_audio_file",
            return_value=(complete_transcript, "en"),
        ):
            result = self.client.post(
                f"/api/recordings/{audio['id']}/transcribe"
            )

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["text"], complete_transcript)
        self.assertEqual(
            self.client.get("/api/recordings").json()[0]["transcript"],
            complete_transcript,
        )

    def test_live_transcription_commits_stable_words_from_rolling_windows(self) -> None:
        sample_rate = 8000
        window = b"\x00\x40" * int(sample_rate * main.LIVE_CHUNK_SECONDS)
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
                            (0.8, 1.0, "there"),
                            (1.4, 1.6, "Watch"),
                            (1.8, 1.95, "this."),
                        ],
                    ),
                    (
                        "Watch this. friend.",
                        "fr",
                        [
                            (0.0, 0.1, "Watch"),
                            (0.4, 0.6, "this."),
                            (1.2, 1.4, "friend."),
                        ],
                    ),
                ],
            ) as transcribe_chunk,
        ):
            with self.client.websocket_connect(
                f"/api/recordings/live?sample_rate={sample_rate}"
            ) as websocket:
                ready = websocket.receive_json()
                self.assertEqual(ready["type"], "status")
                self.assertEqual(ready["window_seconds"], main.LIVE_CHUNK_SECONDS)
                self.assertEqual(ready["overlap_seconds"], main.LIVE_OVERLAP_SECONDS)
                self.assertEqual(
                    ready["unstable_tail_seconds"],
                    main.LIVE_UNSTABLE_TAIL_SECONDS,
                )
                websocket.send_bytes(window)
                first = websocket.receive_json()
                websocket.send_bytes(window)
                second = websocket.receive_json()
                websocket.send_text('{"type":"stop"}')

        self.assertEqual(first["text"], "Hello there Watch")
        self.assertEqual(second["text"], "Hello there Watch this. friend.")
        self.assertEqual(second["language"], "en")
        self.assertEqual(second["segment"], "this. friend.")
        first_prompt = transcribe_chunk.call_args_list[0].args[3]
        second_prompt = transcribe_chunk.call_args_list[1].args[3]
        self.assertIsNone(transcribe_chunk.call_args_list[0].args[2])
        self.assertIsNone(transcribe_chunk.call_args_list[1].args[2])
        self.assertEqual(first_prompt, main.LIVE_MEDICAL_PROMPT)
        self.assertEqual(second_prompt, main.LIVE_MEDICAL_PROMPT)

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
                temperature,
                initial_prompt,
            ):
                self.language = language
                self.vad_filter = vad_filter
                self.word_timestamps = word_timestamps
                self.beam_size = beam_size
                self.temperature = temperature
                self.initial_prompt = initial_prompt
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
        self.assertEqual(model.temperature, 0.0)
        self.assertEqual(model.language, "pt")
        self.assertIn("Clinical note transcription", model.initial_prompt)

    def test_live_transcription_keeps_repeated_words(self) -> None:
        sample_rate = 8000
        window = b"\x00\x40" * int(sample_rate * main.LIVE_CHUNK_SECONDS)
        with (
            patch("main.get_live_whisper_model"),
            patch(
                "main.transcribe_pcm_chunk",
                side_effect=[
                    (
                        "no no",
                        "en",
                        [(0.1, 0.2, "no"), (1.4, 1.6, "no")],
                    ),
                    (
                        "no yes",
                        "en",
                        [(0.0, 0.05, "no"), (0.5, 0.6, "yes")],
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

    def test_live_stop_commits_unstable_tail_before_closing(self) -> None:
        sample_rate = 8000
        window = b"\x00\x40" * int(sample_rate * main.LIVE_CHUNK_SECONDS)
        with (
            patch("main.get_live_whisper_model"),
            patch(
                "main.transcribe_pcm_chunk",
                return_value=(
                    "Hello important word",
                    "en",
                    [
                        (0.1, 0.3, "Hello"),
                        (1.78, 1.84, "important"),
                        (1.86, 1.95, "word"),
                    ],
                ),
            ),
        ):
            with self.client.websocket_connect(
                f"/api/recordings/live?sample_rate={sample_rate}"
            ) as websocket:
                websocket.receive_json()
                websocket.send_bytes(window)
                live_result = websocket.receive_json()
                websocket.send_text('{"type":"stop"}')
                final_result = websocket.receive_json()

        self.assertEqual(live_result["text"], "Hello")
        self.assertEqual(live_result["segment"], "Hello")
        self.assertEqual(live_result["unconfirmed"], "important word")
        self.assertEqual(final_result["text"], "Hello important word")
        self.assertEqual(final_result["segment"], "important word")
        self.assertEqual(final_result["unconfirmed"], "")
        self.assertEqual(final_result["language"], "en")

    def test_silent_live_window_returns_transcript_ack_without_inference(self) -> None:
        sample_rate = 8000
        silence = b"\x00\x00" * int(sample_rate * main.LIVE_CHUNK_SECONDS)
        with (
            patch("main.get_live_whisper_model"),
            patch("main.transcribe_pcm_chunk") as transcribe_chunk,
            patch("main.detect_pcm_language") as detect_language,
        ):
            with self.client.websocket_connect(
                f"/api/recordings/live?sample_rate={sample_rate}"
                "&language_mode=expected&expected_languages=hi,en"
                "&language=hi&prompt_style=hinglish"
            ) as websocket:
                websocket.receive_json()
                websocket.send_bytes(silence)
                response = websocket.receive_json()
                websocket.send_text('{"type":"stop"}')

        self.assertEqual(
            response,
            {
                "type": "transcript",
                "text": "",
                "segment": "",
                "unconfirmed": "",
                "language": "hi",
            },
        )
        transcribe_chunk.assert_not_called()
        detect_language.assert_not_called()

    def test_low_level_live_window_skips_inference(self) -> None:
        sample_rate = 8000
        quiet = b"\x01\x00" * int(sample_rate * main.LIVE_CHUNK_SECONDS)
        with (
            patch("main.get_live_whisper_model"),
            patch("main.transcribe_pcm_chunk") as transcribe_chunk,
        ):
            with self.client.websocket_connect(
                f"/api/recordings/live?sample_rate={sample_rate}"
            ) as websocket:
                websocket.receive_json()
                websocket.send_bytes(quiet)
                response = websocket.receive_json()
                websocket.send_text('{"type":"stop"}')

        self.assertEqual(response["segment"], "")
        transcribe_chunk.assert_not_called()

    def test_live_pcm_chunk_is_wrapped_as_wav(self) -> None:
        def verify_wav(
            audio_path: Path,
            language: str | None,
            initial_prompt: str | None,
        ) -> tuple[str, str, list[tuple[float, float, str]]]:
            self.assertIsNone(language)
            self.assertIsNone(initial_prompt)
            with wave.open(str(audio_path), "rb") as audio_file:
                self.assertEqual(audio_file.getnchannels(), 1)
                self.assertEqual(audio_file.getsampwidth(), 2)
                self.assertEqual(audio_file.getframerate(), 16000)
                self.assertEqual(audio_file.readframes(2), b"\x01\x00" * 2)
            return "Hello", "en", [(0.0, 0.1, "Hello")]

        with patch("main.transcribe_live_audio_file", side_effect=verify_wav):
            result = main.transcribe_pcm_chunk(b"\x01\x00" * 2, 16000)

        self.assertEqual(result, ("Hello", "en", [(0.0, 0.1, "Hello")]))

    def test_live_language_detection_resamples_pcm_to_16khz(self) -> None:
        class FakeModel:
            def detect_language(self, *, audio):
                self.audio = audio
                return "hi", 0.8, [("hi", 0.8), ("en", 0.2)]

        model = FakeModel()
        with patch("main.get_live_whisper_model", return_value=model):
            audio, language, confidence, probabilities = main.detect_pcm_language(
                b"\x01\x00" * 800,
                8000,
            )

        self.assertEqual(len(audio), 1600)
        self.assertEqual(model.audio.shape, audio.shape)
        self.assertEqual((language, confidence), ("hi", 0.8))
        self.assertEqual(probabilities[0], ("hi", 0.8))

    def test_live_websocket_passes_selected_language_to_transcription(self) -> None:
        sample_rate = 8000
        window = b"\x00\x40" * int(sample_rate * main.LIVE_CHUNK_SECONDS)
        with (
            patch("main.get_live_whisper_model"),
            patch(
                "main.transcribe_pcm_chunk",
                return_value=("Olá", "pt", [(0.1, 0.3, "Olá")]),
            ) as transcribe_chunk,
        ):
            with self.client.websocket_connect(
                f"/api/recordings/live?sample_rate={sample_rate}"
                "&language_mode=single&language=pt"
            ) as websocket:
                websocket.receive_json()
                websocket.send_bytes(window)
                self.assertEqual(websocket.receive_json()["language"], "pt")
                websocket.send_text('{"type":"stop"}')

        transcribe_chunk.assert_called_once()
        self.assertEqual(
            transcribe_chunk.call_args.args[1:3],
            (sample_rate, "pt"),
        )
        self.assertIn(
            "Clinical note transcription",
            transcribe_chunk.call_args.args[3],
        )

    def test_live_websocket_hinglish_uses_hindi_and_fixed_prompt(self) -> None:
        sample_rate = 8000
        window = b"\x00\x40" * int(sample_rate * main.LIVE_CHUNK_SECONDS)
        detected_audio = object()
        with (
            patch("main.get_live_whisper_model"),
            patch(
                "main.detect_pcm_language",
                return_value=(
                    detected_audio,
                    "en",
                    0.9,
                    [("en", 0.9), ("hi", 0.08), ("ta", 0.02)],
                ),
            ),
            patch(
                "main.transcribe_live_audio",
                return_value=("Doctor, mujhe bukhar hai.", "hi", []),
            ) as transcribe_audio,
        ):
            with self.client.websocket_connect(
                f"/api/recordings/live?sample_rate={sample_rate}"
                "&language_mode=expected&expected_languages=hi,en"
                "&language=hi&prompt_style=hinglish"
            ) as websocket:
                websocket.receive_json()
                websocket.send_bytes(window)
                response = websocket.receive_json()
                websocket.send_text('{"type":"stop"}')

        self.assertEqual(response["language"], "hi")
        transcribe_audio.assert_called_once_with(
            detected_audio,
            "hi",
            main.LIVE_HINGLISH_PROMPT,
        )

    def test_expected_language_selection_rejects_foreign_language_noise(self) -> None:
        language, confidence = main.select_detected_language(
            [("ta", 0.86), ("hi", 0.09), ("en", 0.05)],
            frozenset({"hi", "en"}),
        )

        self.assertIsNone(language)
        self.assertEqual(confidence, 0.09)

    def test_expected_language_changes_need_two_confident_windows(self) -> None:
        sample_rate = 8000
        window = b"\x00\x40" * int(sample_rate * main.LIVE_CHUNK_SECONDS)
        detections = [
            ("pt", 0.9, [("pt", 0.9), ("fr", 0.05)]),
            ("pt", 0.9, [("pt", 0.9), ("fr", 0.05)]),
            ("fr", 0.9, [("fr", 0.9), ("pt", 0.05)]),
            ("fr", 0.9, [("fr", 0.9), ("pt", 0.05)]),
        ]
        with (
            patch("main.get_live_whisper_model"),
            patch(
                "main.detect_pcm_language",
                side_effect=[
                    (object(), language, confidence, probabilities)
                    for language, confidence, probabilities in detections
                ],
            ),
            patch(
                "main.transcribe_live_audio",
                side_effect=lambda audio, language, prompt: ("", language, []),
            ) as transcribe_audio,
        ):
            with self.client.websocket_connect(
                f"/api/recordings/live?sample_rate={sample_rate}"
                "&language_mode=expected&expected_languages=pt,fr"
            ) as websocket:
                websocket.receive_json()
                responses = []
                for _ in detections:
                    websocket.send_bytes(window)
                    responses.append(websocket.receive_json())
                websocket.send_text('{"type":"stop"}')
                websocket.receive_json()

        self.assertEqual(
            [call.args[1] for call in transcribe_audio.call_args_list],
            ["pt", "pt", "fr"],
        )
        self.assertEqual(
            [response["language"] for response in responses],
            [None, "pt", "pt", "fr"],
        )

    def test_live_websocket_resumes_with_transcript_context_and_language(self) -> None:
        sample_rate = 8000
        window = b"\x00\x40" * int(sample_rate * main.LIVE_CHUNK_SECONDS)
        with (
            patch("main.get_live_whisper_model"),
            patch(
                "main.transcribe_pcm_chunk",
                return_value=("there again", "en", [(1.2, 1.4, "again")]),
            ) as transcribe_chunk,
        ):
            with self.client.websocket_connect(
                f"/api/recordings/live?sample_rate={sample_rate}"
            ) as websocket:
                websocket.receive_json()
                websocket.send_text(
                    json.dumps(
                        {
                            "type": "resume",
                            "transcript": "Hello there",
                            "resume_at_seconds": 2,
                            "language": "en",
                        }
                    )
                )
                websocket.send_bytes(window)
                response = websocket.receive_json()
                websocket.send_text('{"type":"stop"}')

        self.assertEqual(response["segment"], "again")
        self.assertEqual(response["text"], "Hello there again")
        self.assertEqual(
            transcribe_chunk.call_args.args[3],
            main.LIVE_MEDICAL_PROMPT,
        )
        self.assertEqual(transcribe_chunk.call_args.args[2], "en")

    def test_live_websocket_ignores_unknown_query_and_resume_languages(self) -> None:
        sample_rate = 8000
        window = b"\x00\x40" * int(sample_rate * main.LIVE_CHUNK_SECONDS)
        with (
            patch("main.get_live_whisper_model"),
            patch(
                "main.transcribe_pcm_chunk",
                return_value=("Hello there", "en", [(0.1, 0.3, "Hello")]),
            ) as transcribe_chunk,
        ):
            with self.client.websocket_connect(
                f"/api/recordings/live?sample_rate={sample_rate}&language=xx"
            ) as websocket:
                ready = websocket.receive_json()
                websocket.send_text(
                    '{"type":"resume","transcript":"","resume_at_seconds":0,'
                    '"language":"unsupported"}'
                )
                websocket.send_bytes(window)
                response = websocket.receive_json()
                websocket.send_text('{"type":"stop"}')
                final = websocket.receive_json()

        self.assertEqual(ready["status"], "ready")
        self.assertEqual(response["language"], "en")
        self.assertIsNone(transcribe_chunk.call_args.args[2])
        self.assertEqual(final["type"], "transcript")

    def test_live_websocket_initial_resume_keeps_early_words(self) -> None:
        sample_rate = 8000
        window = b"\x00\x40" * int(sample_rate * main.LIVE_CHUNK_SECONDS)
        with (
            patch("main.get_live_whisper_model"),
            patch(
                "main.transcribe_pcm_chunk",
                return_value=("Hello there", "en", [(0.1, 0.3, "Hello")]),
            ),
        ):
            with self.client.websocket_connect(
                f"/api/recordings/live?sample_rate={sample_rate}"
            ) as websocket:
                websocket.receive_json()
                websocket.send_text(
                    '{"type":"resume","transcript":"","resume_at_seconds":0}'
                )
                websocket.send_bytes(window)
                response = websocket.receive_json()
                websocket.send_text('{"type":"stop"}')

        self.assertEqual(response["segment"], "Hello")

    def test_live_initial_prompt_is_static_and_style_specific(self) -> None:
        self.assertEqual(
            main.build_live_initial_prompt(),
            main.LIVE_MEDICAL_PROMPT,
        )
        self.assertEqual(
            main.build_live_initial_prompt("hinglish"),
            main.LIVE_HINGLISH_PROMPT,
        )
        self.assertIn(
            "Doctor, mujhe bukhar hai aur headache ho raha hai.",
            main.build_live_initial_prompt("hinglish"),
        )
        self.assertNotIn(
            "Recent transcript:",
            main.build_live_initial_prompt("hinglish"),
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
            def transcribe(self, audio_path, **kwargs):
                self.audio_path = audio_path
                self.kwargs = kwargs
                segments = [
                    SimpleNamespace(text=" Hello"),
                    SimpleNamespace(text=" world."),
                ]
                info = SimpleNamespace(language="en")
                return iter(segments), info

        model = FakeModel()
        with patch("main.get_whisper_model", return_value=model):
            transcript, language = main.transcribe_audio_file(
                Path(self.temp_dir.name) / "sample.webm",
            )

        self.assertEqual(transcript, "Hello world.")
        self.assertEqual(language, "en")
        self.assertEqual(
            model.kwargs,
            {
                "language": None,
                "vad_filter": True,
                "condition_on_previous_text": False,
                "word_timestamps": True,
                "hallucination_silence_threshold": 2,
                "initial_prompt": None,
            },
        )

    def test_faster_whisper_decoder_accepts_audio(self) -> None:
        audio_path = Path(self.temp_dir.name) / "decoder-smoke.wav"
        with wave.open(str(audio_path), "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(16000)
            audio_file.writeframes(b"\x00\x00" * 1600)

        decoded = decode_audio(str(audio_path))
        self.assertGreater(decoded.size, 0)

    def test_audio_capture_worklet_is_served(self) -> None:
        response = self.client.get("/static/audio-capture-worklet.js")

        self.assertEqual(response.status_code, 200)
        self.assertIn("registerProcessor", response.text)


if __name__ == "__main__":
    unittest.main()
