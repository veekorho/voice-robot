import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000

BLOCK_MS = 30
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)

WINDOW_SECONDS = 5
TRANSCRIBE_EVERY = 2

MODEL_NAME = "small.en"

print("Loading Whisper model...")

model = WhisperModel(
    MODEL_NAME,
    device="cpu",
    compute_type="int8"
)

print("Model loaded.")


rolling_buffer = deque()
buffer_lock = threading.Lock()

def audio_callback(indata, frames, time_info, status):

    if status:
        print(status)

    with buffer_lock:

        rolling_buffer.append(indata.copy())

        total_samples = sum(
            chunk.shape[0]
            for chunk in rolling_buffer
        )

        while total_samples > MAX_SAMPLES:

            removed = rolling_buffer.popleft()

            total_samples -= removed.shape[0]

def get_new_text(previous, current):

    previous = previous.strip()
    current = current.strip()

    if not previous:
        return current

    if current.startswith(previous):
        return current[len(previous):].strip()

    return current

def handle_command(text):

    text = text.lower()

    if "forward" in text:
        print("MOVE FORWARD")

    elif "left" in text:
        print("TURN LEFT")

    elif "right" in text:
        print("TURN RIGHT")

    elif "stop" in text:
        print("STOP")

    elif str(6) in text and str(7) in text:
        print(''' ██████╗   ███████╗
██╔════╝   ╚════██║
██████╗       ██╔╝
██╔══██╗     ██╔╝
██║  ██║    ██╔╝
╚█████╔╝    ██║
 ╚════╝     ╚═╝''')
    else:
        print("UNKNOWN COMMAND")

def transcriber_worker():

    previous_transcript = ""

    while True:

        time.sleep(TRANSCRIBE_EVERY)

        with buffer_lock:

            if not rolling_buffer:
                continue

            audio = np.concentate(
                list(rolling_buffer),
                axis=0
            )

        audio = audio.flatten().astype(np.float32)
        audio /= 32768.0

        try:

            start_time = time.perf_counter()

            segments, info = model.transcribe(

                audio,
                language="en",
                beam_size=5
            )

            current_transcript = " ".join(
                segment.text
                for segment in segments
            ).strip()

            elapsed = (
                time.perf_counter()
                - start_time
            )

            print("\n================================")
            print("Previous")
            print(previous_transcript)

            print("\nCurrent:")
            print(current_transcribe)

            new_text = get_new_text(
                previous_transcribe,
                current_transcribe
            )

            print("\nNew")
            print(new_text)

            print(
                f"\nTranscription time: "
                f"{elapsed:.2f}s"
            )
            print("================================")

            if new_text:
                handle_commad(new_text)

            previous_transcript = current_transcript

        except Exception as e:

            print("Transcription error:", e)

print("Starting transcription thread...")

threading.Thread(
    target=transcriber_worker,
    daemon=True
).start()

print("Listening...")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channel=1,
    dtype="int16",
    blocksize=BLOCK_SIZE,
    callback=audio_callback
):

    try:

        while True:
            time.sleep(1)

    except KeyboardInterrupt:

        print("\nStopped.")

