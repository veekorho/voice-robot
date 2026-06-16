import queue
import tempfile
import os
from collections import deque
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
BLOCK_MS = 30
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)

BUFFER_SECONDS = 3

RMS_THRESHOLD = 500
SILENCE_BLOCKS = 20


audio_queue = queue.Queue()

print("Loading Whisper model...")
model = WhisperModel(
    "small.en",
    device="cpu",
    compute_type="int8"
)

############################################

def callback(indata, frames, time, status):
 if status:
  print(status)

 audio_queue.put(indata.copy())

############################################

def handle_command(text):
 text = text.lower()

 if "forward" in text:
  print("MOVE FORWARD")
 elif "left" in text:
  print("TURN LEFT")
 elif "right" in text:
  print("TURN RIGHT")
 elif "back" in text:
  print("MOVE BACKWARD")
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

###########################################

print("Always listening...")

ROLLING_BLOCKS = int(
 BUFFER_SECONDS * SAMPLE_RATE / BLOCK_SIZE
)

rolling_buffer = deque(maxlen=ROLLING_BLOCKS)
InputStream = sd.InputStream(
 samplerate=SAMPLE_RATE,
 channels=1,
 dtype="int16",
 blocksize=BLOCK_SIZE,
 callback=callback
)
with InputStream:

 recording = False
 silence_count = 0
 speech_buffer = []

 while True:
  block = audio_queue.get()

  rolling_buffer.append(block)

  audio_float = block.astype(np.float32)

  rms = np.sqrt(
   np.mean(audio_float ** 2)
  )

  if rms > RMS_THRESHOLD:

   if not recording:
    print("\nSpeech detected")

    recording = True

    speech_buffer = list(rolling_buffer)

   speech_buffer.append(block)

   silence_count = 0

  elif recording:

   speech_buffer.append(block)

   silence_count += 1

   if silence_count >= SILENCE_BLOCKS:

    recording = False

    InputStream.stop()

    print("Transcribing...")
    
    audio = np.concatenate(

     speech_buffer,
     axis=0
    )

    temp_path = None

    try:

     with tempfile.NamedTemporaryFile(
      suffix=".wav",
      delete=False
     ) as f:

      temp_path = f.name

     write(
      temp_path,
      SAMPLE_RATE,
      audio
     )

     segments, info = model.transcribe(
      temp_path,
      language="en",
      beam_size=5,
      vad_filter=True,
      vad_parameters=dict(
       min_silence_duration_ms=300
      )
     )

     text = " ".join(
      segment.text
      for segment in segments
     ).strip()

     if text:

      print("Heard:", text)

      handle_command(text)

     else:

      print("No speech recognized.")

    except Exception as e:

     print("Error:", e)

    finally:

     if temp_path and os.path.exists(temp_path):
      os.remove(temp_path)

    speech_buffer = []
    silence_count = 0
    InputStream.start()
