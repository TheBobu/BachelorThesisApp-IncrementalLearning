import tensorflow as tf
from tensorflow import keras


class StatsCallback(keras.callbacks.Callback):
    def __init__(self):
        self.model_train_accuracy = []
        self.model_train_loss = []
        self.model_value_accuracy = []
        self.model_value_loss = []
        
    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs["accuracy"]
        val_accuracy = logs["val_accuracy"]
        loss = logs["loss"]
        val_loss = logs["val_loss"]
        self.model_train_accuracy.append(accuracy)
        self.model_train_loss.append(loss)
        self.model_value_accuracy.append(val_accuracy)
        self.model_value_loss.append(val_loss)
