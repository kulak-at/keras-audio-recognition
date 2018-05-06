import pyaudio
import time
import wave
import numpy as np
from mfcc import frames2mfcc
from keras.models import model_from_yaml
from convert_data import get_labels

import matplotlib.pyplot as plt

RECORDINGS_PATH = './recordings/'

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 2048
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = RECORDINGS_PATH + str(time.time()) + '.wav'

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

frames = np.empty([0,])

DESIRED_FRAMES_COUNT = 3*5120 # No idea why


# Loading model

OUTPUT_PATH = './output/'

yaml_file = open(OUTPUT_PATH + 'model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()

model = model_from_yaml(loaded_model_yaml)
model.load_weights(OUTPUT_PATH + 'model.h5')

#
# plt.ion()
# plt.show()
#
# graph, = plt.plot(np.linspace(0, 1, int(DESIRED_FRAMES_COUNT/3)), np.zeros([int(DESIRED_FRAMES_COUNT/3)]))


# def callback(in_data, frame_count, time_info, status):
#     global model, frames
#     data0 = np.frombuffer(in_data, dtype='float32')
#     frames = np.append(frames, data0)
#     if len(frames) >= DESIRED_FRAMES_COUNT:
#         frames = frames[len(frames) - DESIRED_FRAMES_COUNT:]
#         print("Processing now")
#
#
#
# stream = audio.open(
#     format=pyaudio.paFloat32,
#     channels=1,
#     rate=RATE,
#     input=True,
#     frames_per_buffer=CHUNK,
#     stream_callback=callback
# )
#
# stream.start_stream()


try:
    while True:
        data = stream.read(CHUNK)
        data_np = np.fromstring(data, dtype=np.int16) / 38000
        frames = np.append(frames, data_np)
        if len(frames) > DESIRED_FRAMES_COUNT:
            frames = frames[len(frames) - DESIRED_FRAMES_COUNT:]

        if len(frames) == DESIRED_FRAMES_COUNT:
            f = frames[::3]
            # graph.set_ydata(f)
            # plt.draw()
            # plt.pause(0.001)

            m = np.abs(f).mean()
            if m < 0:
                print('-')
            else:
                inp = frames2mfcc(f).reshape(1, 20, 11, 1)
                # print(get_labels()[0][
                #     np.argmax(model.predict(inp))
                # ])
                predicts = model.predict(inp)
                if np.max(predicts) < 0.8:
                    print('.')
                else:
                    print(get_labels()[0][np.argmax(predicts)], np.max(predicts))
        #print(len(frames))
except KeyboardInterrupt:
    pass

# print("Terminating")
# stream.stop_stream()
# stream.close()
# audio.terminate()

# waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# waveFile.setnchannels(CHANNELS)
# waveFile.setsampwidth(audio.get_sample_size(FORMAT))
# waveFile.setframerate(RATE)
# waveFile.writeframes(b''.join(frames))
# waveFile.close()