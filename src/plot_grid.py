import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.models.conv_ae import ConvAE
from src.models.conv_kl_ae import ConvKLAE
from src.models.conv_kl_ae_v1 import ConvKLAEV1
from src.models.conv_kl_ae_v2 import ConvKLAEV2

from src.dataset import Dataset
from src.cwt_generator import CWTGenerator

TEST_DATASET = 'data/dataset/12-28-20T16-12-13$10000'

MODELS = {
    'conv_ae': ConvAE,
    'conv_kl_ae': ConvKLAE,
    'conv_kl_ae_v1': ConvKLAEV1,
    'conv_kl_ae_v2': ConvKLAEV2
}

K = [200, 400, 600, 800, 1000]
M5 = [6000, 6500, 7000, 7500, 8000]

POINTS = list()

for k in K:
    for m5 in M5:
        POINTS.append([m5, k])

K = [1500, 2000, 2500]
M5 = [2000, 3000, 4000, 5000, 6000, 7000]

for k in K:
    for m5 in M5:
        POINTS.append([m5, k])


def get_data(model_name):
    data = list()

    for point in POINTS:
        path_point = 'data/grid/%s/m5_%s_k_%s.npy' % (model_name, point[0], point[1],)

        data.append(np.load(path_point))

    return np.array(data)


def plot(data):
    m5 = data[:, 0]
    k = data[:, 1]

    p_value = data[:, 2]

    plt.contour((m5, k, p_value))
    plt.show()


def main():
    model_path = sys.argv[1].split("@")[4]
    model_name = model_path.split("/")[-2]

    grid_path = 'data/grid/%s' % model_name
    os.makedirs(grid_path, exist_ok=True)

    # data = get_data(model_path, model_type, model_name)
    data = get_data(model_name)
    plot(data)


if __name__ == "__main__":
    main()

