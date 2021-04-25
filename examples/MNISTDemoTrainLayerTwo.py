from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
from keras.datasets import mnist

scaling_factor = 2
clauses = scaling_factor*4000
threshold = scaling_factor*80*100
s = 2.5

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train_transformed = 1-np.load("X_train_transformed.npz")['arr_0']
X_test_transformed = 1-np.load("X_test_transformed.npz")['arr_0']

print(X_train_transformed.shape)

tm = MultiClassTsetlinMachine(clauses, threshold, s, append_negated=False)

print("\nAccuracy over 250 epochs:\n")
max_accuracy = 0.0
for i in range(250):
	start_training = time()
	tm.fit(X_train_transformed, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test_transformed) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
