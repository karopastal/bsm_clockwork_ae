import os
import numpy as np

# import json
# import tensorflow as tf

# from shutil import copy
from datetime import datetime, date
# from matplotlib import pyplot as plt


def normalize(dataset, factor):
    return dataset / factor


def get_ae_base_dir(ae_type, name):
    today = date.today()
    now = datetime.now()
    current_day = today.strftime("%b-%d-%y")
    current_time = now.strftime("%H-%M-%S")

    base_dir = "data/models/%s/%s/%s_T_%s" % (ae_type, name, current_day, current_time,)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    return base_dir


def load_train_bg_data(path_dataset):
    path_train_data = path_dataset + '/train_background.npy'
    return np.load(path_train_data)


def load_test_bg_data(path_dataset):
    path_test_bgs_data = path_dataset + '/test_background.npy'
    return np.load(path_test_bgs_data)