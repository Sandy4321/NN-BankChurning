import numpy as np
import h5py
import util
from model_impl import define_model

if __name__ == "__main__":

	h5file = h5py.File('../gen/dataset.h5', 'r')

	X_train = np.array(h5file['X_train'], dtype=np.float32).T
	X_test = np.array(h5file['X_test'], dtype=np.float32).T
	Y_train = np.array(h5file['Y_train']).T
	Y_test = np.array(h5file['Y_test']).T

	features = X_train.shape[0]
	m = X_train.shape[1]

	layers = [features, 10, 5, 1]

	train = define_model(layers)
	epochs = 10000

	for i in range(epochs):
		cost, W1, W2, W3, b1, b2, b3 = train(X_train, Y_train, 0.01)
		if i % 100 == 0:
			print('Cost on epoch %i: %s' %(i, cost))
