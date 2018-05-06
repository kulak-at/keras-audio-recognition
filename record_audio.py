import pyaudio
import time
import wave

import matplotlib.pyplot as plt
import numpy as np
from mfcc import wav2mfcc

RECORDINGS_PATH = './recordings/'

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = RECORDINGS_PATH + str(time.time()) + '.wav'

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print('recoding ...')

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("Finished recording...")

stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

d = np.fromstring(b''.join(frames), dtype=np.int16) / 38000 # precise value to be found



plt.plot(d)
plt.show()

wav2mfcc(WAVE_OUTPUT_FILENAME)