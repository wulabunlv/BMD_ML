import pickle

import numpy
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

with open('datamrosbmd1103_B1THD', 'rb') as file_handler:
    data = pickle.load(file_handler)
    X, Y = data.get('X', []).values, data.get('Y', []).values


def linear_regression():
    model = Sequential()
    model.add(Dense(23, input_dim=23, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])
    return model


def main(plot=True):
    # fix random seed for reproducibility
    # seed = 7

    # The below is necessary for starting Numpy generated random numbers in a well-defined initial state.
    # numpy.random.seed(seed)
    # The below is necessary for starting core Python generated random numbers in a well-defined state.
    # rn.seed(seed)

    # according to keras documentation, numpy seed should be set before importing keras
    # information regarding setup for obtaining reproducible results using Keras during development in the following link https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

    # The below tf.set_random_seed() will make random number generation in the TensorFlow backend have a well-defined initial state.
    # tf.set_random_seed(seed)

    batch_size = 50
    # num_classes = 1
    # epochs = 50
    number_of_data = X.shape[0]
    number_of_train_data = int(.8 * number_of_data)
    # number_of_test_data = number_of_data - number_of_train_data

    # load dataset
    x_train, x_test = X[:number_of_train_data, :], X[number_of_train_data:, :]
    #mean_train_data = numpy.mean(train_data, axis=0)
    #std_train_data = numpy.std(train_data, axis=0)
    #x_train = (train_data - mean_train_data) / std_train_data  # mean variance normalization
    #x_test = (test_data - mean_train_data) / std_train_data  # mean variance normalization
    y_train, y_test = Y[:number_of_train_data], Y[number_of_train_data:]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
#     print('x_train shape:', x_train.shape)
#     print(x_train.shape[0], 'train samples')
#     print(x_test.shape[0], 'test samples')

    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    model = linear_regression()
    # history = model.fit(x_train, y_train, batch_size=batch_size, epochs=3, verbose=1, validation_data=(x_test, y_test))
    history = model.fit(x_train, y_train, batch_size=batch_size, verbose=1, epochs=100, validation_data=(x_test, y_test))
#     print(history.history.keys())
    score = model.evaluate(x_test, y_test, verbose=0)
#     print('Test loss:', score)

    score = model.evaluate(x_train, y_train, verbose=0)
#     print('Train loss:', score)
    y_pred = model.predict(x_test)

#     print('Mean Squared Error of test: ', mean_squared_error(y_test, y_pred))
#     print('Mean Squared Error of train:', mean_squared_error(y_train, model.predict(x_train)))

#     print('Mean Absolute Error of test: ', mean_absolute_error(y_test, y_pred))
#     print('Mean Absolute Error of train: ', mean_absolute_error(y_train, model.predict(x_train)))

#     print('Coefficient of Determination for test: ', r2_score(y_test, y_pred))
#     print('Coefficient of Determination for train: ', r2_score(y_train, model.predict(x_train)))

    if not plot:
        return history.history['loss'], history.history['val_loss']
    pyplot.plot(history.history['loss'], 'b-')
    pyplot.plot(history.history['val_loss'], 'r-')
    pyplot.title('Mean Squared Error Loss: Linear Regression for Total Hip BMD')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['Train Data', 'Test Data'], loc='upper right')
    pyplot.savefig('reg_B1THD_MSE')
    pyplot.show()


if __name__ == '__main__':
    main()
