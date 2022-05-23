import datetime
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

from IncrementalModel.traincallback import StatsCallback


class Model():
    def __init__(self):
        Model.define_model(self)
        self.custom_stats_callback = StatsCallback()

    def train(self, x_train, y_train, x_test, y_test):
        log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)

        self.model.fit(
            x_train,
            y_train,
            epochs=10,
            validation_data=(x_test, y_test),
            callbacks=[tensorboard_callback, self.custom_stats_callback],
            shuffle=True)

    def define_model(self):
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), padding='same',
                                activation=tf.nn.relu, input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2), strides=2),
            keras.layers.Conv2D(64, (3, 3), padding='same',
                                activation=tf.nn.relu),
            keras.layers.MaxPooling2D((2, 2), strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        optimizer = tf.optimizers.SGD()
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def load_model(self, attempt_nr):
        self.model = keras.models.load_model(f"./saved_models/{attempt_nr}")
        self.model.summary()

    def save_model(self, attempt_nr):
        self.model.save(f"./saved_models/{attempt_nr}")

    def predict(self, img_path):
        img = image.load_img(
            img_path, color_mode="grayscale", target_size=(28, 28))
        img_batch = (np.expand_dims(img, 0))
        prediction = self.model.predict(img_batch)
        print(prediction)
        digit = np.argmax(prediction[0])
        print(digit)
