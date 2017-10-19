from model_impl import define_model
from model_impl import define_predict
import util
import dataset

if __name__ == "__main__":

	Xtrain, Xtest, Ytrain, Ytest = dataset.load_ds_train_test('../gen/dataset.h5')

	features = Xtrain.shape[0]
	layers = [features, 10, 5, 1]

	train = define_model(layers)
	epochs = 50000

	for i in range(epochs):
		cost, W1, W2, W3, b1, b2, b3 = train(Xtrain, Ytrain, 0.1)
		if i % 100 == 0:
			print('Cost on epoch %i: %s' %(i, cost))

	predictor = define_predict(W1, b1, W2, b2, W3, b3)
	
	Ytrain_hat = util.binary(predictor(Xtrain))
	Ytest_hat = util.binary(predictor(Xtest))

	util.perf_report(Ytrain_hat, Ytrain, Ytest_hat, Ytest)

	
	#implement error analysis
	#implement confusion matrix
	#implement 