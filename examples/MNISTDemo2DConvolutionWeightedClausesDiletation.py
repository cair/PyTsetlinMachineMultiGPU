from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D

import numpy as np
from time import time

from keras.datasets import mnist

import cv2

factor = 40

s = 5.0

T = int(factor*25*10)

clauses = int(factor*2000)

ensembles = 10
epochs = 250

batches = 10

resolution = 2

patch_size = 10
image_size = 28

number_of_x_pos_features = image_size - patch_size
number_of_y_pos_features = image_size - patch_size
number_of_content_features = patch_size*patch_size
number_of_features = number_of_x_pos_features + number_of_y_pos_features + number_of_content_features

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train >= 75, 1, 0) 
X_test = np.where(X_test >= 75, 1, 0)

X_test_binary = np.zeros((X_test.shape[0], 28, 28, resolution)).astype(np.uint8)
for i in range(X_test.shape[0]):
    for r in range(1,resolution):
    	kernel = np.ones((r*2+1,r*2+1),np.uint8)
    	X_test_binary[i,:,:,r] = cv2.dilate(X_test[i].astype(np.uint8), kernel, iterations = 1)
print(X_test_binary.shape)

X_train_binary = np.zeros((X_train.shape[0], 28, 28, resolution)).astype(np.uint8)
for i in range(X_train.shape[0]):
    for r in range(1,resolution):
    	kernel = np.ones((r*2+1,r*2+1),np.uint8)
    	X_train_binary[i,:,:,r] = cv2.dilate(X_train[i].astype(np.uint8), kernel, iterations = 1)

f = open("mnist_%.1f_%d_%d_%d_%d.txt" % (s, clauses, T,  patch_size, resolution), "w+")

for e in range(ensembles):
	tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (patch_size, patch_size), number_of_gpus = 16)

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
