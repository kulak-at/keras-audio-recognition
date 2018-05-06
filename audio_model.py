import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from get_train_test import get_train_test
from keras.utils import plot_model

OUTPUT_PATH = 'output/'
EPOCHS = 200

CLASSES_NUM = 30

X_train, X_test, y_train, y_test = get_train_test()

X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)
y_train_hot = to_categorical(y_train,num_classes=CLASSES_NUM)
y_test_hot = to_categorical(y_test,num_classes=CLASSES_NUM)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(CLASSES_NUM, activation='softmax'))

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
# model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
# model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(CLASSES_NUM, activation='softmax'))

model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.30))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.25))
model.add(Dense(CLASSES_NUM, activation='softmax'))

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy']
)

print(X_train.shape, y_train_hot.shape, X_test.shape, y_test_hot.shape)

plot_model(model, to_file=OUTPUT_PATH + 'model.png', show_shapes=True, show_layer_names=True,)

filepath = 'output/partial/part-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks = [checkpoint]

model.fit(
    X_train,
    y_train_hot,
    batch_size=200,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(X_test, y_test_hot),
    callbacks=callbacks
)

# Saving model
print("Saving model")
model_yaml = model.to_yaml()
with open(OUTPUT_PATH + 'model.yaml', 'w') as yaml_file:
    yaml_file.write(model_yaml)
model_json = model.to_json()
with open(OUTPUT_PATH + 'model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights(OUTPUT_PATH + 'model.h5')