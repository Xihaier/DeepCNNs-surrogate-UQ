r"""Field regressor for one to three mapping
Example of Mindlin plate
"""

# Load modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from regressor import FR_25_alpha
from model import FR_model

# load data
train_data = sio.loadmat('plate_1_to_3.mat')
input_x = train_data['input']
output_y = train_data['output']

# split data into training set and testing set
x_train = input_x[:1024, :]
x_test = input_x[1900:, :]
y_train = output_y[:1024, :]
y_test = output_y[1900:, :]

weight_decay = 7e-6
batch_size = 8
learning_rate = 0.005
epochs = 500
eps = 1e-8
kp = 1
anneal_lr_freq = 20
anneal_lr_rate = 0.75

FR_model(x_train, y_train, x_test, y_test, weight_decay, batch_size, learning_rate, epochs, eps, kp, anneal_lr_freq, anneal_lr_rate)
