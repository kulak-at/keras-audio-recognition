import numpy as np
import pyaudio
from queue import Queue


def get_audio_input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        input=True,
        frames_per_buffer=4096,
        input_device_index=0,
        stream_callback=callback)
    return stream




# Queue to communiate between the audio callback and main thread
q = Queue()

run = True

# Data buffer for the input wavform
data = np.zeros(11, dtype='float16')

def callback(in_data, frame_count, time_info, status):
    global run, data
    data0 = np.frombuffer(in_data, dtype='float16')
    if np.abs(data0).mean() < 10:
        print('-')
        return (in_data, pyaudio.paContinue)
    else:
        print('.')
    data = np.append(data,data0)
    if len(data) > 20:
        data = data[-20:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)

stream = get_audio_input_stream(callback)
stream.start_stream()

try:
    while run:
        data = q.get()
        print(data)
except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
    run = False

stream.stop_stream()
stream.close()