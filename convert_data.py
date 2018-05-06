import numpy as np
import os
from keras.utils import to_categorical
from mfcc import wav2mfcc
from tqdm import tqdm

DATA_PATH = './data/'

def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    labels_indices = np.arange(0, len(labels))
    return labels, labels_indices, to_categorical(labels_indices)

def save_data_to_array(path=DATA_PATH, max_pad_len=11):
    labels, _, _ = get_labels(path)

    for label in labels:
        mfcc_vectors = []
        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        print("Saving " + str(label) + " (" + str(len(wavfiles)) + ")")
        for wavfile in tqdm(wavfiles):
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
        print("")
        np.save(label + '.npy', mfcc_vectors)

