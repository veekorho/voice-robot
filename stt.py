from queue import Queue
import numpy as np
import time
import sounddevice as sd
from faster_whisper import WhisperModel
from collections import deque
import threading

# AUDIO SETTINGS
SAMPLE_RATE = 16000
CHANNELS = 1

# continuosly keep a certain amount of seconds in memory
BUFFER_SECONDS = 5
MAX_SAMPLES = SAMPLE_RATE * BUFFER_SECONDS

# transcribe last x seconds every y seconds
WINDOW_SECONDS = 5
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
WAIT_SECONDS = 1

# shared audio buffer
audio_buffer = deque(maxlen=MAX_SAMPLES)
buffer_lock = threading.Lock()  #only one thread at a time can access buffer

# whisper model
model = WhisperModel(
  "tiny.en",
  device="cpu",
  compute_type="int8")


##  AUDIO
def audio_callback(indata, frames, time_info, status):
  if status:
    print(status)
    #print(indata.shape)  #<- should be (xxx, 1)
    samples = indata[:,0]    
    with buffer_lock:
      audio_buffer.extend(samples)
      
def listen():
  with sd.InputStream(samplerate=SAMPLE_RATE, 
                      channels=CHANNELS, 
                      callback=audio_callback,
                      dtype="float32"):
    while True:
      time.sleep(1)
      

## TRANSCRIPTION
def transcribe(q):
  while True:
    time.sleep(WAIT_SECONDS)
    print("start")
    with buffer_lock:
        print("lock")
        #if len(audio_buffer) <= WINDOW_SAMPLES: ## if not enough audio data skip this iteration
            #continue
        audio_chunk = np.array(list(audio_buffer)[-WINDOW_SAMPLES:], dtype=np.float32)

    #give audio data to AI to transcribe 
    print("Transcribing...")
    segments, info = model.transcribe(audio_chunk,
                                      language="en",
                                      beam_size=1
                                     )
    text = " ".join(segment.text for segment in segments)
    
    #remove special signs (.!?,.... ect.) and make everything lowercase
    clean_text = "".join(char for char in text if char.isalnum() or char == " ").lower()
    print(f"[TRANSCRIPT] {clean_text}")

    #add text to queue for the next text to read
    q.put(clean_text)


if __name__ == "__main__":
  print(sd.query_devices())
  lt = threading.Thread(target=listen, daemon=True)
  tt = threading.Thread(target=transcribe, args=(Queue(),), daemon=True)
  lt.start()
  tt.start()


