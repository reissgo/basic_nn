# https://www.youtube.com/watch?v=JdXxaZcQer8

import numpy as np
from random import random
import tensorflow as tf  #  i think this is the CPU version (not GPU)

from sklearn.model_selection import train_test_split


def generate_dataset(num_samples, test_fraction_size):
    x_all_inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(num_samples)])
    y_all_outputs = np.array([[x_single_input[0] + x_single_input[1]] for x_single_input in x_all_inputs])

    x_train, x_test, y_train, y_test = train_test_split(x_all_inputs, y_all_outputs, test_size = test_fraction_size)
    return x_train, x_test, y_train, y_test
