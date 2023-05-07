import pandas
import tensorflow as tf
import numpy as np
import glob


class Network:
    def __init__(self, environment):
        # environment? f8 : f10
        self.environment = environment
        self.directory = 'pomiary/' + self.environment + '/'
        # Ekstrahowanie danych
        self.training_data = pandas.concat((pandas.read_excel(file) for file in glob.glob(self.directory + self.environment + '_stat_*.xlsx')),
                                           ignore_index=True)
        self.mess_data = pandas.concat([self.training_data.pop('data__coordinates__x'), self.training_data.pop('data__coordinates__y')], axis=1)
        self.ref_data = pandas.concat([self.training_data.pop('reference__x'), self.training_data.pop('reference__y')], axis=1)
        self.validation_data = pandas.concat((pandas.read_excel(file) for file in glob.glob(self.directory + self.environment + '_random_*.xlsx')),
                                             ignore_index=True)
        self.val_mess_data = pandas.concat([self.validation_data.pop('data__coordinates__x'), self.validation_data.pop('data__coordinates__y')],
                                           axis=1)
        self.val_ref_data = pandas.concat([self.validation_data.pop('reference__x'), self.validation_data.pop('reference__y')], axis=1)
        # Tworzenie właściwej sieci
        self.network = tf.keras.models.Sequential()
        self.network.add(tf.keras.layers.Dense(128, activation='relu'))
        self.network.add(tf.keras.layers.Dense(64, activation='relu'))
        self.network.add(tf.keras.layers.Dense(32, activation='relu'))
        self.network.add(tf.keras.layers.Dense(16, activation='relu'))
        self.network.add(tf.keras.layers.Dense(8, activation='relu'))
        self.network.add(tf.keras.layers.Dense(2, activation='sigmoid'))

        # Compile network using Adam the optimizer
        self.network.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss=tf.keras.losses.MeanSquaredError(),
                        metrics=['accuracy'])

        # Fit (train) network, calculate it's accuracy using validation data set
        self.network.fit(np.asarray(self.mess_data), np.asarray(self.ref_data), epochs=150, batch_size=512,
                    validation_data=(np.asarray(self.val_mess_data), np.asarray(self.val_ref_data)))

    def test(self, case):
        test_data = pandas.read_excel(self.directory + self.environment + '_' + case + '.xlsx')
        test_mess_data = pandas.concat([test_data.pop('data__coordinates__x'), test_data.pop('data__coordinates__y')], axis=1)
        test_ref_data = pandas.concat([test_data.pop('reference__x'), test_data.pop('reference__y')], axis=1)

        # Evaluate network to get loss value and metrics values
        self.network.evaluate(np.asarray(test_mess_data), np.asarray(test_ref_data), batch_size=512)
        weights = self.network.layers[0].get_weights()[0]

        # Get output for desired measurements using trained network
        result = self.network.predict(np.asarray(test_mess_data))

        return result, weights
