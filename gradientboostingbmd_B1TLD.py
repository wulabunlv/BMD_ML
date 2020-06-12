import pickle
from matplotlib import pyplot
import numpy

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

with open('datamrosbmd1103_B1TLD', 'rb') as file_handler:
    data = pickle.load(file_handler)
    X, Y = data.get('X', []).values, data.get('Y', []).values


def create_model():
    return RandomForestRegressor(n_estimators=35, verbose=1)


def main(plot=True):
    # fix random seed for reproducibility
    # seed = 7
    # The below is necessary for starting Numpy generated random numbers in a well-defined initial state.
    # numpy.random.seed(seed)
    # The below is necessary for starting core Python generated random numbers in a well-defined state.
    # random.seed(seed)

    # The below tf.set_random_seed() will make random number generation in the TensorFlow backend have a well-defined initial state.
    # tf.set_random_seed(seed)
    # Y = label_binarize(Y, classes=[0,1])

    # batch_size = 120
    # num_classes = 2
    # epochs = 15

    number_of_data = X.shape[0]
    number_of_train_data = int(.8 * number_of_data)
    # number_of_test_data = number_of_data - number_of_train_data

    # load dataset for MLP
    x_train, x_test = X[:number_of_train_data, :], X[number_of_train_data:, :]
    #mean_train_data = numpy.mean(train_data, axis=0)
    #std_train_data = numpy.std(train_data, axis=0)
    #x_train = (train_data - mean_train_data) / std_train_data  # mean variance normalization
    #x_test = (test_data - mean_train_data) / std_train_data  # mean variance normalization
    y_train, y_test = Y[:number_of_train_data], Y[number_of_train_data:]

    model = create_model()

    validatescores = cross_val_score(model, x_train, y_train)
#     print(validatescores, ' ALL TRY')

    history = model.fit(x_train, y_train)
#     print(history.feature_importances_)
#     print(history, ' HISTORY')
    # y_pred = model.predict(x_test)

    params = {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
    gbr = GradientBoostingRegressor(**params)
    gbr.fit(x_train, y_train)

    test_score = numpy.zeros((params['n_estimators'],), dtype=numpy.float64)
    mse = mean_squared_error(y_test, gbr.predict(x_test))

#     print('Mean Square Error of test: ', mean_squared_error(y_test, gbr.predict(x_test)))
#     print('Mean Square Error of train: ', mean_squared_error(y_train, gbr.predict(x_train)))

#     print('Coefficient of Determination for test: ', r2_score(y_test, gbr.predict(x_test)))
#     print('Coefficient of Determination for train: ', r2_score(y_train, gbr.predict(x_train)))

    for i, y_pred in enumerate(gbr.staged_predict(x_test)):
        test_score[i] = gbr.loss_(y_test, y_pred)
    if not plot:
        return gbr.train_score_, test_score
    pyplot.figure()
    pyplot.title('Mean Square Error Loss: Gradient Boosting for Total Spine BMD')
    pyplot.plot(numpy.arange(params['n_estimators']) + 1, gbr.train_score_, 'b-', label='Training')
    pyplot.plot(numpy.arange(params['n_estimators']) + 1, test_score, 'r-', label='Test')
    pyplot.legend(loc='upper right')
    pyplot.legend(loc='upper right')
    pyplot.xlabel('epoch')
    pyplot.ylabel('Loss')
    pyplot.savefig('gb_B1TLD_MSE')
    pyplot.show()


if __name__ == '__main__':
    main()
