from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D

import numpy as np
from time import time

from keras.datasets import mnist

factor = 40

s = 5.0

T = int(factor*25*10)

ensembles = 10
epochs = 250

batches = 10

patch_size = 10

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train >= 75, 1, 0) 
X_test = np.where(X_test >= 75, 1, 0)

batch_size= X_train.shape[0]//batches

f = open("mnist_%.1f_%d_%d_%d.txt" % (s, int(factor*2000), T,  patch_size), "w+")

for e in range(ensembles):
	tm = MultiClassConvolutionalTsetlinMachine2D(int(factor*2000), T, s, (patch_size, patch_size), clause_drop_p = 0.1, feature_drop_p = 0.1, number_of_gpus = 16)

	for i in range(epochs):
		for b in range(batches):
			start_training = time()
			tm.fit(X_train[b*batch_size:(b+1)*batch_size], Y_train[b*batch_size:(b+1)*batch_size], epochs=1, incremental=True)
			stop_training = time()
			print("\tBatch #%d %.2f" % (b, stop_training-start_training))

		start_testing = time()
		result_test = 100*(tm.predict(X_test) == Y_test).mean()
		stop_testing = time()

		result_train = 100*(tm.predict(X_train) == Y_train).mean()

		print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
		print("%d %d %.2f %.2f %.2f %.2f" % (e, i, result_train, result_test, stop_training-start_training, stop_testing-start_testing), file=f)
		f.flush()
f.close()
