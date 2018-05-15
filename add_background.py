import os
from scikits.audiolab import wavread, wavwrite
from convert_data import get_labels
import random


BACKGROUND_NOISE_FILE = "./office.wav"
INPUT_DIRECTORY = "./input"
OUTPUT_DIRECTORY = "./output"

SAMPLES_PER_ORYGINAL = 10

def addBackgroundNoise():
    background, fs, enc = wavread(BACKGROUND_NOISE_FILE)
    background = background[:,0]
    labels, _, _ = get_labels(INPUT_DIRECTORY)
    for label in labels:
        generateSamples(label, background)

def generateSamples(label, background):
    print 'Saving ' + label
    output_folder = OUTPUT_DIRECTORY + '/' + label + '/'
    input_folder = INPUT_DIRECTORY + '/' + label + '/'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    wavfiles = [input_folder + '/' + wavfile for wavfile in os.listdir(input_folder)]
    print(wavfiles)
    nr = 1
    for file in wavfiles:
        data, fs, _ = wavread(file)
        data_len = len(data)
        for i in range(0, SAMPLES_PER_ORYGINAL):
            start = random.randint(0, len(background) - data_len)
            ratio = 0.25 + random.random() * 0.5 # range 0.25 - 0.75
            sample = (1 - ratio) * data + ratio * background[start:start+data_len]
            print 'Saving ' + label + '/' + str(nr) + '.wav'
            wavwrite(sample, output_folder + str(nr) + '.wav', fs=fs)
            nr += 1




addBackgroundNoise()