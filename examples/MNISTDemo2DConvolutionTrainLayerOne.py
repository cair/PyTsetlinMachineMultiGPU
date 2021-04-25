from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train >= 75, 1, 0) 
X_test = np.where(X_test >= 75, 1, 0) 

tm = MultiClassConvolutionalTsetlinMachine2D(4000, 10*100, 10.0, (10, 10))

print("\nAccuracy over 20 epochs:\n")
max_accuracy = 0.0
for i in range(20):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	if result > max_accuracy:
		max_accuracy = result
		max_ta_state = tm.get_state()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))

tm.set_state(max_ta_state)

start_testing = time()
result = 100*(tm.predict(X_test) == Y_test).mean()
stop_testing = time()

print("Accuracy: %.2f%% Testing: %.2fs" % (result, stop_testing-start_testing))

print("\nTransforming datasets")
start_transformation = time()
X_train_transformed = tm.transform(X_train)
X_test_transformed = tm.transform(X_test)
stop_transformation = time()
print("Transformation time: %.fs" % (stop_transformation - start_transformation))

print("Saving transformed datasets")
np.savez_compressed("X_train_transformed.npz", X_train_transformed)
np.savez_compressed("X_test_transformed.npz", X_test_transformed)