import numpy as np
import src.models.utils as model_utils
# import tensorflow.python.keras.backend as backend
# import tensorflow as tf

from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model, Sequential
# from keras.optimizers import Adam
from keras.callbacks import CSVLogger
# from keras import regularizers


# def kl_divergence(rho, rho_hat):
#     return rho * backend.log(rho) - \
#            rho * backend.log(rho_hat) + \
#            (1 - rho) * backend.log(1 - rho) - \
#            (1 - rho) * backend.log(1 - rho_hat)


# class SparsityRegularizer(regularizers.Regularizer):
#
#     def __init__(self, rho=0.1, beta=3):
#         self.rho = rho
#         self.beta = beta
#
#     def __call__(self, x):
#         regularization = backend.constant(0., dtype=x.dtype)
#         rho_hat = backend.mean(x, axis=0)
#         regularization += self.beta * tf.math.reduce_sum(kl_divergence(self.rho, rho_hat))
#
#         return regularization
#
#     def get_config(self):
#         return {'rho': float(self.rho), 'beta': float(self.beta)}


class ConvAE:
    def __init__(self,
                 path_model='',
                 path_dataset='',
                 ae_type='conv_ae',
                 name='conv_ae',
                 optimizer='adam',
                 shape=(60, 56),
                 beta=3,
                 rho=0.005):

        self.ae_type = ae_type
        self.name = name
        self.path_model = path_model

        if path_model != '':
            self.path_autoencoder = self.path_model + '/autoencoder.h5'
            self.path_summary = self.path_model + '/summary.txt'
            self.path_loss_progress = self.path_model + '/training.log'

            self.autoencoder_model = load_model(self.path_autoencoder)

            # self.path_dataset = self.dataset_config['PATH_DATASET']
            # self.shape = self.dataset_config['ORIGINAL_SHAPE']

        else:
            self.path_dataset = path_dataset
            self.base_dir = model_utils.get_ae_base_dir(self.ae_type, self.name)
            self.path_autoencoder = self.base_dir + '/autoencoder.h5'
            self.path_summary = self.base_dir + '/summary.txt'
            self.path_csv_logger = self.base_dir + '/training.log'

            self.shape = shape
            self.autoencoder_model = self.build_model()
            self.autoencoder_model.compile(loss='mse', optimizer=optimizer)

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(128,
                         kernel_size=(3, 3),
                         activation='elu',
                         padding='same',
                         input_shape=(self.shape[0], self.shape[1], 1)))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='elu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='elu', padding='same'))

        model.add(Flatten())
        model.add(Dense(40, activation='elu'))
        """ Encoded layer """
        model.add(Dense(20, activation='elu'))
        model.add(Dense(40, activation='elu'))
        model.add(Dense(int(np.prod(self.shape) / 16 * 128), activation='elu'))
        model.add(Reshape((int(self.shape[0] / 4), int(self.shape[1] / 4), 128)))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='elu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='elu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(1, kernel_size=(3, 3), activation='elu', padding='same'))

        return model

    def load_train_data(self):
        train_bg = model_utils.load_train_bg_data(self.path_dataset)
        test_bg = model_utils.load_test_bg_data(self.path_dataset)

        factor = -1 * np.log(0.01)
        norm_train_bg = model_utils.normalize(train_bg, factor)
        norm_test_bg = model_utils.normalize(test_bg, factor)

        np.reshape(norm_train_bg,
                   (norm_train_bg[0],
                    norm_train_bg[1],
                    norm_train_bg[2],
                    1))

        np.reshape(norm_test_bg,
                   (norm_test_bg[0],
                    norm_test_bg[1],
                    norm_test_bg[2],
                    1))

        return norm_train_bg, norm_test_bg

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

    # def load_test_data(self, signal_id=1):
    #     test_bgs_data = model_utils.load_test_bgs_data(self.path_dataset)
    #     test_signal_data = model_utils.load_test_signal_data(self.path_dataset, signal_id=signal_id)
    #
    #     factor = -1 * np.log(0.01)
    #     norm_test_bgs_data = model_utils.normalize(test_bgs_data, factor)
    #     norm_test_signal_data = model_utils.normalize(test_signal_data, factor)
    #
    #     return norm_test_bgs_data, norm_test_signal_data

    # def predict(self, x_test):
    #     predictions = self.autoencoder_model.predict(x_test)
    #
    #     return predictions
    #
    # def eval_model(self, name, signal_id=1, file_name=''):
    #     losses = {}
    #     test_bgs_data, test_signal_data = self.load_test_data(signal_id=signal_id)
    #
    #     predict_bgs_test = self.predict(test_bgs_data)
    #     predict_signal_test = self.predict(test_signal_data)
    #
    #     losses['test_bgs_data'] = test_bgs_data
    #     losses['test_signal_data'] = test_signal_data
    #     losses['predict_bgs_test'] = predict_bgs_test.reshape(losses['test_bgs_data'].shape)
    #     losses['predict_signal_test'] = predict_signal_test.reshape(losses['test_signal_data'].shape)
    #
    #     model_utils.print_predictions_loss(losses=losses)
    #
    #     title = '%s Background' % (name,)
    #     model_utils.plot_prediction(self.autoencoder_model,
    #                                 test_bgs_data[0:3],
    #                                 self.shape,
    #                                 title,
    #                                 file_name=[file_name, '_bg'])
    #
    #     title = '%s Background + Signal' % (name,)
    #     model_utils.plot_prediction(self.autoencoder_model,
    #                                 test_signal_data[0:3],
    #                                 self.shape,
    #                                 title,
    #                                 file_name=[file_name, '_bg_signal'])
    #
    # def create_loss_distribution(self, name, signal_id=1, file_name=''):
    #     test_bgs_data, test_signal_data = self.load_test_data(signal_id=signal_id)
    #
    #     predict_bgs_test = self.predict(test_bgs_data)
    #     predict_signal_test = self.predict(test_signal_data)
    #
    #     test_bgs_distribution = model_utils.loss_distribution(test_bgs_data,
    #                                                           predict_bgs_test.reshape(test_bgs_data.shape))
    #
    #     test_signal_distribution = model_utils.loss_distribution(test_signal_data,
    #                                                              predict_signal_test.reshape(test_signal_data.shape))
    #     model_utils.plot_histogram(test_bgs_distribution.numpy(),
    #                                test_signal_distribution.numpy(),
    #                                name,
    #                                file_name=file_name)

    def summary(self):
        return self.autoencoder_model.summary()

    # def plot_progress(self, title='', file_name=''):
    #     if self.path_model != '':
    #         model_utils.plot_progress(self.path_loss_progress, title=title, file_name=file_name)
    #     else:
    #         print('error, load model first')


""" Versions """


def conv_ae_1():
    return "lala 11"


def conv_ae_2():
    return "lala 22"


def conv_ae_3():
    return "lala 32"
