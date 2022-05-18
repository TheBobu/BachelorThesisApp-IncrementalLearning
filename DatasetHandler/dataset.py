import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np

class Dataset():
    input_train_shape = (60000, 28, 28, 1)
    input_test_shape = (10000, 28, 28, 1)
    output_train_shape = (60000,)
    output_test_shape = (10000,)
    
    def __init__(self):
        Dataset.load_data(self)
    
    def load_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        self.x_train = x_train.reshape(Dataset.input_train_shape)
        self.y_train = to_categorical(
            y_train.reshape(Dataset.output_train_shape))
        self.x_test = x_test.reshape(Dataset.input_test_shape)
        self.y_test = to_categorical(y_test.reshape(Dataset.output_test_shape))
        self.output_layer_size = len(np.unique(y_train))
        
       
    def shuffle_data(self):
        pass 
        
    def get_random_data(self, number_of_items):
        pass