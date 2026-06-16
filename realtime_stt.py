import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
RECORD_SECONDS = 3
AUDIO_FILE = "command.wav"

model = WhisperModel(

    "base",
    device="cpu",
    compute_type="int8"
)

def record_audio():
    print("Listening... speak now")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()
    write(AUDIO_FILE, SAMPLE_RATE, audio)
    print("Audio recorded")

def transcribe_audio():
    segments, _ = model.transcribe(AUDIO_FILE)

    text = " ".join(segment.text for segment in segments).lower().strip()

    print("Heard:", text)
    return text

def handle_command(text):
    if "forward" in text:
        print("MOVE FORWARD")

    elif "back" in text:
        print("MOVE BACKWARD")

    elif "left" in text:
        print("TURN LEFT")

    elif "right" in text:
        print("TURN RIGHT")

    elif "stop" in text:
        print("STOP")

    else:
        print("No valid command")

print("Voice robot started. Say commands...")

while True:
    try:
        record_audio()
        text = transcribe_audio()
        handle_command(text)

        time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n Stoppig robot")
        break
