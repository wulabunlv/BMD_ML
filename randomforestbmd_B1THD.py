import pickle

import numpy
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

with open('datamrosbmd1103_B1THD', 'rb') as file_handler:
    data = pickle.load(file_handler)
    X, Y = data.get('X', []).values, data.get('Y', []).values

# fix random seed for reproducibility
# seed = 7

# The below is necessary for starting Numpy generated random numbers in a well-defined initial state.
# numpy.random.seed(seed)
# The below is necessary for starting core Python generated random numbers in a well-defined state.
# random.seed(seed)

# The below tensorflow.set_random_seed() will make random number generation in the TensorFlow backend have a well-defined initial state.
# tensorflow.set_random_seed(seed)
# Y = label_binarize(Y, classes=[0,1])

batch_size = 120
num_classes = 2

number_of_data = X.shape[0]
number_of_train_data = int(.8 * number_of_data)
number_of_test_data = number_of_data - number_of_train_data

x_train, x_test = X[:number_of_train_data, :], X[number_of_train_data:, :]
# mean_train_data = numpy.mean(train_data, axis=0)
# std_train_data = numpy.std(train_data, axis=0)
# x_train = (train_data - mean_train_data) / std_train_data  # mean variance normalization
# x_test = (test_data - mean_train_data) / std_train_data  # mean variance normalization
y_train, y_test = Y[:number_of_train_data], Y[number_of_train_data:]

# RANDOM_STATE = 42
n_estimators = 100


# override the RandomForestRegressor library
class RandomForestRegressorCustom(RandomForestRegressor):
    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix instead, shape = (n_samples, n_samples_fitted], where n_samples_fitted is the number of samples used in the fitting for the estimator.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """

        return mean_squared_error(y, self.predict(X))


def create_model(epoch):
    return RandomForestRegressorCustom(n_estimators=epoch, random_state = 42, warm_start=True, oob_score=True, max_features='sqrt', max_depth=4)


def main(plot=True):
    epoch = 100
    model = create_model(epoch)
    model.fit(x_train, y_train)
    model.score(x_test, y_test)
#     print(model.score(x_test, y_test), ' SOE')

    train_score, test_score = [], []

    for i in range(epoch):
        model = create_model(i + 1)
        model.fit(x_train, y_train)
        train_score.append(model.score(x_train, y_train))
        test_score.append(model.score(x_test, y_test))
#     print(test_score, ' TEST SCORE')
#     print(train_score, ' TRAIN SCORE')

#     print('Mean Square Error of test: ', mean_squared_error(y_test, model.predict(x_test)))
#     print('Mean Square Error of train: ', mean_squared_error(y_train, model.predict(x_train)))

#     print('Coefficient of Determination for test: ', r2_score(y_test, model.predict(x_test)))
#     print('Coefficient of Determination for train: ', r2_score(y_train, model.predict(x_train)))

    if not plot:
        return train_score, test_score
    pyplot.plot(range(epoch), train_score, 'b-')
    pyplot.plot(range(epoch), test_score, 'r-')
    pyplot.title('Mean Squared Error Loss: Random Forest for Total Hip BMD')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['Train Data', 'Test Data'], loc='upper right')
    pyplot.savefig('rf_B1THD_MSE')
    pyplot.show()


if __name__ == '__main__':
    main()
