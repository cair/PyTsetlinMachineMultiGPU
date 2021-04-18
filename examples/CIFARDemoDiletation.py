from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
from keras.preprocessing.image import ImageDataGenerator

from keras.datasets import cifar10

import cv2

epochs = 100
ensembles = 10

factor = 40
clauses = int(4000*factor)
T = int(75*10*factor)
s = 20.0
patch_size = 8
resolution = 2
step_size = 3
number_of_state_bits = 9

labels = [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

Y_train=Y_train.reshape(Y_train.shape[0])

Y_test=Y_test.reshape(Y_test.shape[0])

X_test_binary = np.zeros((X_test.shape[0], 32, 32, resolution*3)).astype(np.uint8)
for i in range(X_test.shape[0]):
    for r in range(resolution):
        kernel = np.ones((r*2+1,r*2+1),np.uint8)
        for j in range(3):
            X_test_binary[i,:,:,r*3+j] = cv2.dilate(cv2.adaptiveThreshold(X_test[i,:,:,j].astype(np.uint8), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2), kernel, iterations = 1)
print(X_test_binary.shape)

X_train_binary = np.zeros((X_train.shape[0], 32, 32, resolution*3)).astype(np.uint8)
for i in range(X_train.shape[0]):
    for r in range(resolution):
        kernel = np.ones((r*2+1,r*2+1),np.uint8)
        for j in range(3):
            X_train_binary[i,:,:,r*3+j] = cv2.dilate(cv2.adaptiveThreshold(X_train[i,:,:,j].astype(np.uint8), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2), kernel, iterations = 1)

datagen = ImageDataGenerator(
    rotation_range=0,
    horizontal_flip=False,
    width_shift_range=0,
    height_shift_range=0
    #zoom_range=0.3
    )
datagen.fit(X_train)

# Introduce augmented data here

f = open("cifar10_%.1f_%d_%d_%d_%d_%d.txt" % (s, clauses, T,  patch_size, resolution, number_of_state_bits), "w+")

for e in range(ensembles):
    tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (patch_size, patch_size), number_of_state_bits=number_of_state_bits, number_of_gpus=16)

    for i in range(epochs):
        start_training = time()
        tm.fit(X_train_binary, Y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result_test = 100*(tm.predict(X_test_binary) == Y_test).mean()
        stop_testing = time()

        result_train = 100*(tm.predict(X_train_binary) == Y_train).mean()
        print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
        print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing), file=f)
        f.flush()
f.close()

