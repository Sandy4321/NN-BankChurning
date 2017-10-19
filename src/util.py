import inspect
import sys
import numpy as np

def raiseNotDefined():
	file = inspect.stack()[1][1]
	line = inspect.stack()[1][2]
	method = inspect.stack()[1][3]

	print('\n\n*** Method not implemented: %s at line %s of %s\n\n' % (method, line, file))
	sys.exit(1)

def raiseError(message):
	file = inspect.stack()[1][1]
	line = inspect.stack()[1][2]
	method = inspect.stack()[1][3]

	print('\n\n*** Error %s at line %s of %s: %s\n\n' % (method, line, file, message))
	sys.exit(1)

def binary(Y_pred):
	"""
	
	Converts the input to binary values

	"""
	pred = np.zeros((1, Y_pred.shape[1]))

	for i in range(Y_pred.shape[1]):
		if Y_pred[0, i] > 0.5:
			pred[0,i] = 1

	return pred

def accuracy(Y_pred, Y_actual):
	"""
		
	Computes the accuracy of the prediction based on the labels

	"""
	return 100 - np.mean(np.abs(Y_pred - Y_actual))*100

def recall(Y_pred, Y_actual):
	"""

	Computes the recall of the model - of all 1's how many the classifier got right

	"""
	counter = 0.
	correct = 0.

	for i in range(Y_actual.shape[1]):

		if Y_actual[0][i] == 1:
			counter = counter + 1

			if Y_pred[0][i] == 1:
				correct = correct + 1

	if counter == 0:
		return 0

	rec = (correct / counter)*100
	return rec

def f1_score(accuracy, recall):
	"""

	Computes F1 Score of the model

	"""
	if accuracy == 0 or recall == 0:
		return 0
	else:
		return 200 / ((100/accuracy) + (100/recall))

def perf_report(Ytrain_hat, Ytrain, Ytest_hat, Ytest):
	"""

	Prints the neural network performance report

	"""
	acc_train = accuracy(Ytrain_hat, Ytrain)
	acc_test = accuracy(Ytest_hat, Ytest)

	rec_train = recall(Ytrain_hat, Ytrain)
	rec_test = recall(Ytest_hat, Ytest)

	f1_train = f1_score(acc_train, rec_train)
	f1_test = f1_score(acc_test, rec_test)

	print('\n\n---------------------------------------')
	print(' MODEL PERFORMANCE')
	print('---------------------------------------')

	print('\n-Train Set-----------------------------\n')
	print('* Accuracy = %3.2f' % (acc_train) + '%')
	print('* Recall   = %3.2f' % (rec_train) + '%')
	print('* F1 Score = %3.2f' % (f1_train) + '%')
	print('---------------------------------------')


	print('\n\n-Test Set------------------------------\n')
	print('* Accuracy = %3.2f' % (acc_test) + '%')
	print('* Recall   = %3.2f' % (rec_test) + '%')
	print('* F1 Score = %3.2f' % (f1_test) + '%')
	print('---------------------------------------')

