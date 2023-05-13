import math

import pandas
import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plot


class Network:
    def __init__(self, environment):
        # environment? f8 : f10
        self.environment = environment
        self.directory = 'pomiary/' + self.environment + '/'
        # Ekstrahowanie danych
        self.training_data = pandas.concat((pandas.read_excel(file) for file in glob.glob(self.directory + self.environment + '_stat_*.xlsx')),
                                           ignore_index=True)
        self.mess_data = pandas.concat([self.training_data.pop('data__coordinates__x'), self.training_data.pop('data__coordinates__y')], axis=1)
        for record in self.mess_data:  # jebane kurwa nulle
            for index, coordinate in enumerate(self.mess_data[record]):
                if pandas.isnull(coordinate):
                    self.mess_data[record][index] = (self.mess_data[record][index - 1] + self.mess_data[record][index + 1]) / 2
        self.mess_data = (self.mess_data.astype('float32') + 3000) / 13000  # gdzieś wyczytałem, że sequential lubi liczby z zakresu [0 ; 1]
        self.ref_data = pandas.concat([self.training_data.pop('reference__x'), self.training_data.pop('reference__y')], axis=1)
        for record in self.ref_data:
            for index, coordinate in enumerate(self.ref_data[record]):
                if pandas.isnull(coordinate):
                    self.ref_data[record][index] = (self.ref_data[record][index - 1] + self.ref_data[record][index + 1]) / 2
        self.ref_data = (self.ref_data.astype('float32') + 3000) / 13000
        self.validation_data = pandas.concat((pandas.read_excel(file) for file in glob.glob(self.directory + self.environment + '_random_*.xlsx')),
                                             ignore_index=True)
        for record in self.validation_data:
            for index, coordinate in enumerate(self.validation_data[record]):
                if pandas.isnull(coordinate):
                    self.validation_data[record][index] = (self.validation_data[record][index - 1] + self.validation_data[record][index + 1]) / 2
        self.validation_data = (self.validation_data.astype('float32') + 3000) / 13000
        self.val_mess_data = pandas.concat([self.validation_data.pop('data__coordinates__x'), self.validation_data.pop('data__coordinates__y')],
                                           axis=1)
        for record in self.val_mess_data:
            for index, coordinate in enumerate(self.val_mess_data[record]):
                if pandas.isnull(coordinate):
                    self.val_mess_data[record][index] = (self.val_mess_data[record][index - 1] + self.val_mess_data[record][index + 1]) / 2
        self.val_mess_data = (self.val_mess_data.astype('float32') + 3000) / 13000
        self.val_ref_data = pandas.concat([self.validation_data.pop('reference__x'), self.validation_data.pop('reference__y')], axis=1)
        for record in self.val_ref_data:
            for index, coordinate in enumerate(self.val_ref_data[record]):
                if pandas.isnull(coordinate):
                    self.val_ref_data[record][index] = (self.val_ref_data[record][index - 1] + self.val_ref_data[record][index + 1]) / 2
        self.val_ref_data = (self.val_ref_data.astype('float32') + 3000) / 13000
        # Tworzenie właściwej sieci
        self.network = tf.keras.models.Sequential()  # najprostszy model sieci
        self.network.add(tf.keras.layers.Dense(128, activation='relu')) # zgodnie z teorią neuronów powinno być coraz mniej
        self.network.add(tf.keras.layers.Dense(64, activation='relu'))
        self.network.add(tf.keras.layers.Dense(32, activation='relu'))
        self.network.add(tf.keras.layers.Dense(16, activation='relu'))
        self.network.add(tf.keras.layers.Dense(8, activation='relu'))
        self.network.add(tf.keras.layers.Dense(2, activation='sigmoid'))  # 2 ponieważ przewidujemy 2 rzeczy, x i y

        # Compile network using Adam the optimizer
        self.network.compile(optimizer=tf.keras.optimizers.Adam(), # TODO: to trzeba ogarnąć co robi
                             loss=tf.keras.losses.MeanSquaredError(),
                             metrics=['accuracy'])

        # Fit (train) network, calculate it's accuracy using validation data set
        self.network.fit(np.asarray(self.mess_data), np.asarray(self.ref_data), epochs=150, batch_size=256, # batch_size - częstość wyboru wag
                         validation_data=(np.asarray(self.val_mess_data), np.asarray(self.val_ref_data)))

    def test(self, case):
        test_data = pandas.read_excel(self.directory + self.environment + '_' + case + '.xlsx')
        test_mess_data = pandas.concat([test_data.pop('data__coordinates__x'), test_data.pop('data__coordinates__y')], axis=1)
        # plot.scatter(np.asarray(test_mess_data)[:, 0], np.asarray(test_mess_data)[:, 1], color='red')
        return_mess = test_mess_data
        test_mess_data = (test_mess_data.astype('float32') + 3000) / 13000  # ten sam przedział wynikowy, co w danych testowych
        test_ref_data = pandas.concat([test_data.pop('reference__x'), test_data.pop('reference__y')], axis=1)
        # plot.plot(np.asarray(test_ref_data)[:, 0], np.asarray(test_ref_data)[:, 1], color='green')
        return_ref = test_ref_data
        test_ref_data = (test_ref_data.astype('float32') + 3000) / 13000

        # Evaluate network to get loss value and metrics values
        self.network.evaluate(np.asarray(test_mess_data), np.asarray(test_ref_data), batch_size=256)
        weights = self.network.layers[0].get_weights()[0]

        # Get output for desired measurements using trained network
        result = self.network.predict(np.asarray(test_mess_data))
        result = (result * 13000) - 3000

        # plot.scatter(np.asarray(result)[:, 0], np.asarray(result)[:, 1], color='blue')
        # plot.show()
        return np.asarray(return_mess), np.asarray(return_ref), np.asarray(result), weights

    # Obliczanie dystrybuanty
    def error(self, mess, ref, result):
        mess_errors = np.sqrt((mess[:, 0] - ref[:, 0]) * (mess[:, 0] - ref[:, 0]) + (mess[:, 1] - ref[:, 1]) * (mess[:, 1] - ref[:, 1]))
        result_errors = np.sqrt((result[:, 0] - ref[:, 0]) * (result[:, 0] - ref[:, 0]) + (result[:, 1] - ref[:, 1]) * (result[:, 1] - ref[:, 1]))
        mess_number_of_errors = []
        result_number_of_errors = []
        for i in range(len(ref)): # w każdej iteracji zliczamy ile jest błędów, które są mniejsze wartością od i
            mess_number_of_errors.append(np.count_nonzero(mess_errors < i) / len(ref)) # wartości dystrybuanty dla poszczególnych wartości
            result_number_of_errors.append(np.count_nonzero(result_errors < i) / len(ref))
        plot.plot(range(len(ref)), mess_number_of_errors, color='red')
        plot.plot(range(len(ref)), result_number_of_errors, color='blue')
        plot.show()





