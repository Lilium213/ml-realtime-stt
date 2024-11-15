import numpy as np
import pyaudio
import whisper
import threading


def transcribe(frames):
    # print("Transcribing...")
    joined_frames = b"".join(frames)
    numpy_array = np.frombuffer(joined_frames, dtype=np.float32)
    result = model.transcribe(numpy_array)
    print(result["text"])


CHUNK = 1024
RATE = 16000
RECORD_SECONDS = 1

model = whisper.load_model("turbo")

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
stream.start_stream()

try:
    while True:
        # print("Listening...")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        threading.Thread(target=transcribe, args=(frames,)).start()
        #transcribe(frames)
except KeyboardInterrupt:
    print("Terminating...")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
