import queue
from collections import deque
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

command_queue = queue.Queue()
audio_queue = queue.Queue()

############################################

#Audio input handling
def callback(indata, frames, time, status):
 if status:
  print(status)

 audio_queue.put(indata.copy())

###########################################

#Transcription process
def transcribe(q):
 
 print("Loading Whisper model...")
 model = WhisperModel( #Load transcription model
     "tiny.en", #model (i.e small, small.en, medium...)
     device="cpu", #cuda, cpu or auto
     compute_type="int8"
 )
 
 #Config values

 SAMPLE_RATE = 16000
 BLOCK_MS = 30
 BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)

 BUFFER_SECONDS = 1.5 #Amount of recording in buffer

 RMS_THRESHOLD = 1500 #Adjust recording start volume
 
 SILENCE_BLOCKS = 35 #Adjust how much total silence is needed before recording ends
 
 print("Always listening...")
 
 ROLLING_BLOCKS = int(
  BUFFER_SECONDS * SAMPLE_RATE / BLOCK_SIZE
 )

 rolling_buffer = deque(maxlen=ROLLING_BLOCKS)
 
 with sd.InputStream(
  samplerate=SAMPLE_RATE,
  channels=1,
  dtype="int16",
  blocksize=BLOCK_SIZE,
  callback=callback
 ):
  print("with inputstream")

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

   if rms > RMS_THRESHOLD: #Wait until loud enough audio, then begin recording

    if not recording:
     print("\nSpeech detected")

     recording = True

     speech_buffer = list(rolling_buffer)

    speech_buffer.append(block)

    silence_count = 0

   elif recording:

    speech_buffer.append(block)

    silence_count += 1

    if silence_count >= SILENCE_BLOCKS: #Wait for long enough silence, then begin transcription and stop recording

     recording = False

     print("Transcribing...")
    
     audio = np.concatenate(
      speech_buffer,
      axis=0
     )

     audio = audio.flatten().astype(np.float32)
     audio /= 32768.0 #Magic number that makes the program work
    
     try:

      segments, info = model.transcribe(
       audio,
       language="en",
       beam_size=5, #Number of beams used for beam search
       condition_on_previous_text=False, #Whether or not the model will use context from previous transcriptions for making new ones
       hotwords="execute, off, stop", #The model will be more likely to recognize words listed here
       vad_filter=True, #Filters out silences of certain length from input data; define the length with min_silence_duration_ms
       vad_parameters=dict(
        min_silence_duration_ms=300
       )
      )

      text = " ".join(
       segment.text
       for segment in segments
      ).strip()

      if text:
       clean_text = text.lower()
       clean_text = clean_text.replace(".", "")
       clean_text = clean_text.replace("!", "")
       clean_text = clean_text.replace(",", "")
       
       q.put(clean_text)
    
      else:
      
       print("No speech recognized.")

     except Exception as e:

      print("Error:", e)

     speech_buffer = []
     silence_count = 0
