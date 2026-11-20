"""Run from your clinical-notes folder:
    python bench_live2.py data/recordings/<any recording>.webm
Compares live-window cost for: tiny vs tiny.en, word timestamps on/off, 1 s vs 2 s windows.
(tiny.en is downloaded the first time, so you need internet for that one run.)"""
import os
import sys
import time

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

full = decode_audio(sys.argv[1], sampling_rate=16000)
cores = os.cpu_count() or 4
print(f"CPU cores: {cores}")


def bench(label, model_name, seconds, word_timestamps):
    audio = full[: 16000 * seconds]
    model = WhisperModel(model_name, device="cpu", compute_type="int8",
                         cpu_threads=cores, num_workers=1)

    def run():
        segments, _ = model.transcribe(
            audio, language="en", beam_size=1, temperature=0.0,
            word_timestamps=word_timestamps,
        )
        return " ".join(s.text.strip() for s in segments)

    run()  # warm-up
    times, text = [], ""
    for _ in range(7):
        start = time.perf_counter()
        text = run()
        times.append(time.perf_counter() - start)
    times.sort()
    print(f"{label:<44} median {times[3] * 1000:5.0f} ms  worst {times[-1] * 1000:5.0f} ms | {text[:45]!r}")


bench("tiny    2 s window, word timestamps (now)", "tiny", 2, True)
bench("tiny    2 s window, no word timestamps", "tiny", 2, False)
bench("tiny    1 s window, word timestamps", "tiny", 1, True)
bench("tiny.en 2 s window, word timestamps", "tiny.en", 2, True)
bench("tiny.en 2 s window, no word timestamps", "tiny.en", 2, False)
print("\nYour stride is 1.5 s, so median must stay well under 1500 ms.")
