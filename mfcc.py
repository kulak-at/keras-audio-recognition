import numpy as np
import librosa

import matplotlib.pyplot as plt


def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    # print(wave)
    # plt.plot(wave)
    # plt.show()
    return frames2mfcc(wave, max_pad_len)



def frames2mfcc(frames, max_pad_len=11):
    mfcc = librosa.feature.mfcc(frames, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc
