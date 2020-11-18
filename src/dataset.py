import os
import logging
import numpy as np

from datetime import datetime

BASEDIR = 'data/dataset'


class Dataset:
    def __init__(self,
                 cwt_generator,
                 train_size=25000,
                 test_bg_size=5000,
                 test_signal_size=5000):

        self.cwt_generator = cwt_generator
        self.train_size = train_size
        self.test_bg_size = test_bg_size
        self.test_signal_size = test_signal_size

        self.path = self.__generate_path()
        self.path_log = self.path + '/progress.log'
        self.path_train = self.path + '/train_background.npy'
        self.path_test_background = self.path + '/test_background.npy'
        self.path_test_signal = self.path + '/test_signal'

        self.__init_logger()
        logging.info('Init dataset')

    @staticmethod
    def current_datetime():
        return datetime.now().strftime("%m-%d-%yT%H-%M-%S")

    def __generate_path(self):
        path = '%s/%s$%s' % (BASEDIR,
                             Dataset.current_datetime(),
                             int(self.train_size))

        os.makedirs(path, exist_ok=True)

        return path

    def __init_logger(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] - %(message)s',
            filename=self.path_log)

        logging.info('init logging')

    def build_train_dataset(self):
        logging.info('START: building train dataset')

        arrays = []

        for i in range(self.train_size):
            record = self.cwt_generator.yield_background()
            arrays.append(record)

        dataset = np.stack(arrays, axis=0)
        np.save(self.path_train, dataset)

        logging.info('DONE: building train dataset')

    def build_test_bg_dataset(self):
        logging.info('START: building test background dataset')

        arrays = []

        for i in range(self.test_bg_size):
            record = self.cwt_generator.yield_background()
            arrays.append(record)

        dataset = np.stack(arrays, axis=0)
        np.save(self.path_test_background, dataset)

        logging.info('DONE: building test background dataset')

    def build_test_signal_dataset(self, m_5, k):
        logging.info('START: building test signal m_5_%s k_%s dataset' % (m_5, k,))

        self.cwt_generator.mount_signal(m_5=m_5, k=k)
        path_signal = self.path_test_signal + '__m_5_%s_k_%s.npy' % (m_5, k,)

        arrays = []

        for i in range(self.test_signal_size):
            record = self.cwt_generator.yield_background_signal()
            arrays.append(record)

        dataset = np.stack(arrays, axis=0)
        np.save(path_signal, dataset)

        logging.info('DONE: building test signal dataset')
