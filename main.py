import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
# Импортируем библиотеку keras с различными модулями
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D, Input, LSTM, Reshape
from keras import regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import brier_score_loss as bsl
from keras.layers import RNN as Keras_RNN


# Функция для построения графиков обучения сетей
def history_plot(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()


# Функция для построения графиков
def plot_graf(test_set, predicted_BTC_price, df_test):
    plt.figure(figsize=(25, 15), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    # test_set = df_test[30:].values
    plt.plot(test_set, color='red', label='Настоящая стоимость BTC')
    plt.plot(predicted_BTC_price, color='blue', label='Предсказаная стоимость BTC')
    plt.title('BTC Price Prediction', fontsize=40)
    df_test = df_test.reset_index()
    x = df_test.index
    labels = df_test['date']
    plt.xticks(x, labels, rotation='vertical')

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)

    plt.xlabel('Time', fontsize=40)
    plt.ylabel('График BTC/USD', fontsize=40)
    plt.legend(loc=2, prop={'size': 25})

    plt.show()


# Рекурентная сеть LSTM
def RNN(X_train, y_train):
    X_train = X_train[10:]
    y_train = y_train[10:]
    X_train = X_train.reshape(int(len(X_train) / 15), 75, 1)
    y_train = y_train.reshape(int(len(y_train) / 15), 75)
    y_train = y_train[:, 50:]

    # Инициализируем последовательность
    model = Sequential()
    # Добаляем входной слой в LSTM слой из 10 нейронов
    model.add(LSTM(units=100, activation='tanh', input_shape=[75, 1]))
    # Добаляем выходной слой
    model.add(Dense(25))
    model.add(Activation('relu'))

    # Сообираем нейронную сеть
    model.compile(optimizer='adam',
                  loss='mse', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
    checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=1, save_best_only=True)
    # Обучаем сеть
    history = model.fit(X_train, y_train, batch_size=100, epochs=90, verbose=0, validation_split=0.15,
                        callbacks=[reduce_lr, checkpointer], shuffle=True)
    history_plot(history)
    model.save('models/RNN_Without_Reg')
    # model = load_model('models/RNN_Without_Reg')

    return model


# Сверточная сеть
def CNN(X_train, y_train):
    X_train = X_train[10:]
    y_train = y_train[10:]
    X_train = X_train.reshape(int(len(X_train) / 15), 75, 1)
    y_train = y_train.reshape(int(len(y_train) / 15), 75)
    y_train = y_train[:, 50:]

    # Инициализируем последовательность
    model = Sequential()
    # Добавляем Сверточный слой
    model.add(Convolution1D(input_shape=[75, 1], kernel_size=10, strides=6, padding='same', activation='relu', filters=50))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Convolution1D(kernel_size=7, strides=3, padding='same', activation='relu', filters=30))
    model.add(Flatten())

    model.add(Dense(25))
    model.compile(optimizer='adam',
                  loss='mse', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
    checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=0, save_best_only=True)
    history = model.fit(x=X_train, y=y_train, batch_size=25, epochs=100, shuffle=True,
                        callbacks=[reduce_lr, checkpointer], validation_split=0.15)
    #
    history_plot(history)

    model.save('models/CNN3')
    # model = load_model('models/CNN3')

    return model


# Многослойная сеть
def MLP(X_train, y_train, count):
    X_train = X_train[10:]
    y_train = y_train[10:]
    X_train = X_train.reshape(int(len(X_train) / 15), 75)
    y_train = y_train.reshape(int(len(y_train) / 15), 75)
    y_train = y_train[:, 50:]

    # Инициализируем последовательность
    model = Sequential()
    # Добавляем слой из 500 пресептронов
    model.add(Dense(200, input_dim=75))
    # Делаем дропаут
    model.add(Dropout(0.15))
    # Добавляем слой из 250 пресептронов
    model.add(Dense(100))
    model.add(Dropout(0.15))
    model.add(Dense(25))
    model.add(Activation('linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
    checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=1, save_best_only=True)
    history = model.fit(X_train, y_train, batch_size=10, epochs=120, verbose=0, validation_split=0.15,
                        shuffle=True, callbacks=[reduce_lr, checkpointer])
    history_plot(history)
    model.save('models/MLP_Without_Reg1')
    # model = load_model('models/MLP_Without_Reg1')

    return model


def predicted_BTC_price_func(regressor, inputs, name, data_frame):
    # inputs = inputs[:30]
    if name is 'MLP':
        inputs = np.reshape(inputs, (int(len(inputs) / 15), 75))
        output = regressor.predict(inputs)
    if name is 'CNN':
        inputs = np.reshape(inputs, (int(len(inputs) / 15), 75, 1))
        output = regressor.predict(inputs)
    if name is 'RNN':
        inputs = np.reshape(inputs, (int(len(inputs) / 15), 75, 1))
        output = regressor.predict(inputs)
    inputs = inputs[1:, :25]
    output = output[:-1]
    inputs = np.reshape(inputs, (15, 5))
    output = np.reshape(output, (15, 5))

    predicted_BTC_price = sc.inverse_transform(output)
    test_set = sc.inverse_transform(inputs)
    test_set = test_set[:, 4]
    predicted_BTC_price = predicted_BTC_price[:, 4]
    # Строим графики
    plot_graf(test_set, predicted_BTC_price, data_frame[-15:])
    mse_nn = mse(test_set, predicted_BTC_price)

    test_set_bsl, predicted_BTC_price_bsl = to_percent_arr(test_set, predicted_BTC_price)
    bsl_nn = bsl(y_true=test_set_bsl, y_prob=predicted_BTC_price_bsl)
    return mse_nn, bsl_nn


def to_percent_arr(test_set_, predicted_price):
    for i in range(0, len(test_set_)):
        percent = predicted_price[i] / test_set_[i]
        if percent > 1:
            predicted_price[i] = 2 - percent
        else:
            predicted_price[i] = percent
        test_set_[i] = 1
    return test_set_, predicted_price


if __name__ == '__main__':
    df = pd.read_csv('data/coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27.csv')

    df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
    df = df.drop(['Volume_(BTC)', 'Volume_(Currency)', 'Timestamp'], axis=1)
    group = df.groupby('date')
    # Делаем предобработку данных
    list_of_values = list([group['Open'].first(), group['High'].max(), group['Low'].min(), group['Close'].last(),
                           group['Weighted_Price'].mean()])
    one_day = pd.DataFrame(data=list_of_values)
    one_day = one_day.T

    np_one_day = one_day.values
    # Для валидации берем 60 дней относительно них мы будем строить прогноз
    prediction_days = 60

    training_set = np_one_day[:-prediction_days]
    test_set = np_one_day[-prediction_days:]

    # Масштабируем данные для того, чтобы было проще обучать нейронные сети
    sc = MinMaxScaler()
    training_set = sc.fit_transform(training_set)
    X_train = training_set[:-1]
    y_train = training_set[1:]

    inputs = test_set

    inputs = sc.transform(inputs)

    # name_of_nn =  'MLP' or 'CNN' or 'RNN'
    name_of_nn = 'CNN'
    mse_list = list()
    bsl_list = list()
    # Обучение рекурентной сети
    if False:
        regressor = RNN(X_train, y_train)
        mse_nn, bsl_nn = predicted_BTC_price_func(regressor, inputs, 'RNN', df[-60:])
        mse_list.append(mse_nn)
        bsl_list.append(bsl_nn)
    # Обучение Сверточной сети
    if True:
        regressor = CNN(X_train, y_train)
        mse_nn, bsl_nn = predicted_BTC_price_func(regressor, inputs, 'CNN', df[-60:])
        mse_list.append(mse_nn)
        bsl_list.append(bsl_nn)
    # Обучение Многослойного Пресептрона
    if False:
        regressor = MLP(X_train, y_train, 20)
        mse_nn, bsl_nn = predicted_BTC_price_func(regressor, inputs, 'MLP', df[-60:])
        mse_list.append(mse_nn)
        bsl_list.append(bsl_nn)
    print(bsl_list)
    print(mse_list)

    # predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)

    # Берем поледнюю 4 строку со средней стоймостью актива
    # test_set = test_set[:, 4]
    # predicted_BTC_price = predicted_BTC_price[:, 4]

    # Строим графики
    # plot_graf(test_set, predicted_BTC_price, df[-60:])

    # Подсчитываем среднеквадратичное отклонение
    # mse_nn = mse(test_set, predicted_BTC_price)

    # print(mse_nn)

    # test_set_bsl, predicted_BTC_price_bsl = to_percent_arr(test_set, predicted_BTC_price)
    # bsl_nn = bsl(y_true=test_set_bsl, y_prob=predicted_BTC_price_bsl)
    #
    # print(bsl_nn)

    plot_model(regressor,to_file=""+name_of_nn+".png",show_shapes=True)
