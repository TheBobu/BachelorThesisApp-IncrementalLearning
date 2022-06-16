import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import os.path
from os import path

class Dataset():
    input_train_shape = (60000, 28, 28, 1)
    input_test_shape = (10000, 28, 28, 1)
    output_train_shape = (60000,)
    output_test_shape = (10000,)

    def __init__(self):
        self.load_data()
        self.max_number_of_rand_images = 100

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        print(type(x_train),type(y_train))
        self.x_train = x_train.reshape(Dataset.input_train_shape)
        self.y_train = to_categorical(
            y_train.reshape(Dataset.output_train_shape))
        self.x_test = x_test.reshape(Dataset.input_test_shape)
        self.y_test = to_categorical(y_test.reshape(Dataset.output_test_shape))
        self.output_layer_size = len(np.unique(y_train))

    def get_data(self, number_of_items, offset):
        return (self.x_train[offset:(offset+number_of_items)], self.y_train[offset:(offset+number_of_items)])
    
    def geneate_random_images(self):
        if path.exists('random_dataset') != True:
            os.mkdir('random_dataset')
        for i in range(0, self.max_number_of_rand_images):
            Z = np.random.random((28,28))   # Test data
            norm = plt.Normalize(vmin=Z.min(), vmax=Z.max())
            cmap = plt.cm.get_cmap(name='gray')
            image = cmap(norm(Z))
            if os.path.exists(f'random_dataset/{i}.png'):
                os.remove(f'random_dataset/{i}.png')
            plt.imsave(f'random_dataset/{i}.png', image)


# if __name__ == "__main__":
#     set = Dataset()
#     a="aaa"
#     print(type(a))
#     a=([],[])
#     print(type(set.get_data(10,20)))