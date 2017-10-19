import numpy as np
import h5py

def load_ds_train_test(filepath):
	"""

	Loads train and test sets to numpy arrays

	"""
	h5file = h5py.File(filepath, 'r')

	X_train = np.array(h5file['X_train'], dtype=np.float32).T
	X_test = np.array(h5file['X_test'], dtype=np.float32).T
	Y_train = np.array(h5file['Y_train']).T
	Y_test = np.array(h5file['Y_test']).T

	return X_train, X_test, Y_train, Y_test

def load_ds_train_dev_test(filepath):
	"""

	Loads train and test sets to numpy arrays

	"""
	h5file = h5py.File(filepath, 'r')

	X_train = np.array(h5file['X_train'], dtype=np.float32).T
	X_dev = np.array(h5file['X_dev'], dtype=np.float32).T
	X_test = np.array(h5file['X_test'], dtype=np.float32).T
	Y_train = np.array(h5file['Y_train']).T
	Y_dev = np.array(h5file['Y_dev']).T
	Y_test = np.array(h5file['Y_test']).T

	return X_train, X_dev, X_test, Y_train, Y_dev, Y_test