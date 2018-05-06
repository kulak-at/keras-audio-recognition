from keras.models import model_from_yaml
from mfcc import wav2mfcc
from convert_data import get_labels
import numpy as np

OUTPUT_PATH = './output/'

yaml_file = open(OUTPUT_PATH + 'model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()

model = model_from_yaml(loaded_model_yaml)
model.load_weights(OUTPUT_PATH + 'model.h5')

# FIXME: think about model.compile if needed

# sample = wav2mfcc('./data/right/0a7c2a8d_nohash_0.wav')

sample = wav2mfcc('./recordings/1525038319.2531478.wav')

sample_reshaped = sample.reshape(1, 20, 11, 1)
print(get_labels()[0][
    np.argmax(model.predict(sample_reshaped))
      ])
