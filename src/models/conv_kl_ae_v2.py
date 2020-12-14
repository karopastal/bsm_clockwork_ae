import json
import numpy as np
import src.models.utils as model_utils
import tensorflow as tf
import tensorflow.python.keras.backend as backend

from keras import regularizers
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model, Sequential
from keras.callbacks import CSVLogger


def kl_divergence(rho, rho_hat):
    return rho * backend.log(rho) - \
           rho * backend.log(rho_hat) + \
           (1 - rho) * backend.log(1 - rho) - \
           (1 - rho) * backend.log(1 - rho_hat)


class SparsityRegularizer(regularizers.Regularizer):

    def __init__(self, rho=0.1, beta=3):
        self.rho = rho
        self.beta = beta

    def __call__(self, x):
        regularization = backend.constant(0., dtype=x.dtype)
        rho_hat = backend.mean(x, axis=0)
        regularization += self.beta * tf.math.reduce_sum(kl_divergence(self.rho, rho_hat))

        return regularization

    def get_config(self):
        return {'rho': float(self.rho), 'beta': float(self.beta)}


class ConvKLAEV2:
    def __init__(self,
                 path_model='',
                 path_dataset='',
                 ae_type='conv_kl_ae_v2',
                 name='conv_kl_ae_v2',
                 optimizer='adam',
                 shape=(60, 56),
                 beta=3,
                 rho=0.005):

        self.name = name
        self.ae_type = ae_type
        self.path_model = path_model

        if path_model != '':
            self.path_dataset = path_dataset
            self.path_autoencoder = self.path_model + '/autoencoder.h5'
            self.path_summary = self.path_model + '/summary.txt'
            self.path_loss_progress = self.path_model + '/training.log'

            self.shape = shape

            custom_objects = {'SparsityRegularizer': SparsityRegularizer}

            self.autoencoder_model = load_model(self.path_autoencoder,
                                                custom_objects=custom_objects)

            self.autoencoder_model = load_model(self.path_autoencoder)

        else:
            self.path_dataset = path_dataset
            self.base_dir = model_utils.get_ae_base_dir(self.ae_type, self.name)
            self.path_autoencoder = self.base_dir + '/autoencoder.h5'
            self.path_summary = self.base_dir + '/summary.txt'
            self.path_csv_logger = self.base_dir + '/training.log'

            self.rho = rho
            self.beta = beta
            self.shape = shape
            self.__save_config()

            self.autoencoder_model = self.build_model()
            self.autoencoder_model.compile(loss='mse', optimizer=optimizer)

    def __save_config(self):
        config = dict()

        config['base_dir'] = self.base_dir
        config['path_dataset'] = self.path_dataset
        config['path_csv_logger'] = self.path_csv_logger
        config['path_summary'] = self.path_summary
        config['shape'] = self.shape
        config['rho'] = self.rho
        config['beta'] = self.beta

        filename = self.base_dir + '/config.json'

        with open(filename, 'w') as outfile:
            json.dump(config, outfile, indent=4)

        outfile.close()

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(128,
                         kernel_size=(3, 3),
                         activation='relu',
                         padding='same',
                         input_shape=(self.shape[0], self.shape[1], 1)))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))

        """ Encoded layer """
        model.add(Dense(128,
                        activation='sigmoid',
                        activity_regularizer=SparsityRegularizer(self.rho,
                                                                 self.beta)))

        model.add(Dense(1024, activation='relu'))
        model.add(Dense(int(np.prod(self.shape) / 16 * 128), activation='elu'))
        model.add(Reshape((int(self.shape[0] / 4), int(self.shape[1] / 4), 128)))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(1, kernel_size=(3, 3), activation='elu', padding='same'))

        return model

    def load_train_data(self):
        train_bg = model_utils.load_train_bg_data(self.path_dataset)
        test_bg = model_utils.load_test_bg_data(self.path_dataset)

        factor = -1 * np.log(0.01)
        norm_train_bg = model_utils.normalize(train_bg, factor)
        norm_test_bg = model_utils.normalize(test_bg, factor)

        # norm_train_bg = train_bg
        # norm_test_bg = test_bg

        reshape_norm_train_bg = np.reshape(norm_train_bg, (
            norm_train_bg.shape[0],
            norm_train_bg.shape[1],
            norm_train_bg.shape[2],
            1))

        reshape_norm_test_bg = np.reshape(norm_test_bg, (
            norm_test_bg.shape[0],
            norm_test_bg.shape[1],
            norm_test_bg.shape[2],
            1))

        return reshape_norm_train_bg, reshape_norm_test_bg

    def train_model(self, epochs=5, batch_size=64):
        train_bg, test_bg = self.load_train_data()

        csv_logger = CSVLogger(self.path_csv_logger)

        self.autoencoder_model.fit(train_bg, train_bg,
                                   epochs=epochs,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   validation_data=(test_bg, test_bg),
                                   callbacks=[csv_logger])

        self.autoencoder_model.save(self.path_autoencoder)

        with open(self.path_summary, 'w') as fh:
            self.autoencoder_model.summary(print_fn=lambda x: fh.write(x + '\n'))

    def load_model(self):
        self.autoencoder_model = load_model(self.path_autoencoder)

    def load_test_data(self, m_5=6000, k=1000):
        test_bgs_data = model_utils.load_test_bg_data(self.path_dataset)

        test_signal_data = model_utils.load_test_signal_data(self.path_dataset,
                                                             m_5=m_5,
                                                             k=k)

        factor = -1 * np.log(0.01)

        # factor = 1
        norm_test_bgs_data = model_utils.normalize(test_bgs_data, factor)
        norm_test_signal_data = model_utils.normalize(test_signal_data, factor)

        reshape_norm_test_bg = np.reshape(norm_test_bgs_data, (
            norm_test_bgs_data.shape[0],
            norm_test_bgs_data.shape[1],
            norm_test_bgs_data.shape[2],
            1))

        reshape_norm_train_bg = np.reshape(norm_test_signal_data, (
            norm_test_signal_data.shape[0],
            norm_test_signal_data.shape[1],
            norm_test_signal_data.shape[2],
            1))

        return reshape_norm_test_bg, reshape_norm_train_bg

    def predict(self, x_test):
        predictions = self.autoencoder_model.predict(x_test)

        return predictions

    def eval_model(self, name, m_5=6000, k=1000, file_name=''):
        losses = {}

        test_bgs_data, test_signal_data = self.load_test_data(m_5=m_5, k=k)

        predict_bgs_test = self.predict(test_bgs_data)
        predict_signal_test = self.predict(test_signal_data)

        losses['test_bgs_data'] = test_bgs_data
        losses['test_signal_data'] = test_signal_data
        losses['predict_bgs_test'] = predict_bgs_test.reshape(losses['test_bgs_data'].shape)
        losses['predict_signal_test'] = predict_signal_test.reshape(losses['test_signal_data'].shape)

        model_utils.print_predictions_loss(losses=losses)

        # title = '%s Background' % (name,)
        # model_utils.plot_prediction(self.autoencoder_model,
        #                             test_bgs_data[0:3],
        #                             self.shape,
        #                             title,
        #                             file_name=[file_name, '_bg'])
        #
        # title = '%s Background + Signal' % (name,)
        # model_utils.plot_prediction(self.autoencoder_model,
        #                             test_signal_data[0:3],
        #                             self.shape,
        #                             title,
        #                             file_name=[file_name, '_bg_signal'])

    def create_loss_distribution(self, name, m_5=6000, k=1000, file_name=''):
        test_bgs_data, test_signal_data = self.load_test_data(m_5=m_5, k=k)

        predict_bgs_test = self.predict(test_bgs_data)
        predict_signal_test = self.predict(test_signal_data)

        test_bgs_distribution = model_utils.loss_distribution(test_bgs_data,
                                                              predict_bgs_test.reshape(test_bgs_data.shape))

        test_signal_distribution = model_utils.loss_distribution(test_signal_data,
                                                                 predict_signal_test.reshape(test_signal_data.shape))
        model_utils.plot_histogram(test_bgs_distribution.numpy(),
                                   test_signal_distribution.numpy(),
                                   name,
                                   file_name=file_name)

    def summary(self):
        return self.autoencoder_model.summary()

    def plot_progress(self, title='', file_name=''):
        if self.path_model != '':
            model_utils.plot_progress(self.path_loss_progress, title=title, file_name=file_name)
        else:
            print('error, load model first')


"""
__Versions__:

1. datasets: 5k, 25k
2. rho: 
3. beta: 

"""

DS_5K = 'data/dataset/12-09-20T17-23-10$5000'
DS_25K = 'data/dataset/11-18-20T23-18-18$25000'


def conv_kl_ae_v2_1():
    rho = 0.05
    beta = 3
    optimizer = 'adam'

    conv_ae = ConvKLAEV2(path_dataset=DS_5K,
                         name='conv_kl_ae_v2_1',
                         rho=rho,
                         beta=beta,
                         optimizer=optimizer)

    conv_ae.train_model(epochs=100, batch_size=64)


def conv_kl_ae_v2_2():
    rho = 0.05
    beta = 3
    optimizer = 'adam'

    conv_ae = ConvKLAEV2(path_dataset=DS_25K,
                         name='conv_kl_ae_v2_2',
                         rho=rho,
                         beta=beta,
                         optimizer=optimizer)

    conv_ae.train_model(epochs=100, batch_size=64)


def conv_kl_ae_v2_3():
    rho = 0.01
    beta = 3
    optimizer = 'adam'

    conv_ae = ConvKLAEV2(path_dataset=DS_5K,
                         name='conv_kl_ae_v2_3',
                         rho=rho,
                         beta=beta,
                         optimizer=optimizer)

    conv_ae.train_model(epochs=100, batch_size=64)


def conv_kl_ae_v2_4():
    rho = 0.01
    beta = 3
    optimizer = 'adam'

    conv_ae = ConvKLAEV2(path_dataset=DS_25K,
                         name='conv_kl_ae_v2_4',
                         rho=rho,
                         beta=beta,
                         optimizer=optimizer)

    conv_ae.train_model(epochs=100, batch_size=64)


def conv_kl_ae_v2_5():
    rho = 0.25
    beta = 3
    optimizer = 'adam'

    conv_ae = ConvKLAEV2(path_dataset=DS_5K,
                         name='conv_kl_ae_v2_5',
                         rho=rho,
                         beta=beta,
                         optimizer=optimizer)

    conv_ae.train_model(epochs=100, batch_size=64)


def conv_kl_ae_v2_6():
    rho = 0.25
    beta = 3
    optimizer = 'adam'

    conv_ae = ConvKLAEV2(path_dataset=DS_25K,
                         name='conv_kl_ae_v2_6',
                         rho=rho,
                         beta=beta,
                         optimizer=optimizer)

    conv_ae.train_model(epochs=100, batch_size=64)
