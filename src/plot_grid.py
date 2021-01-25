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

NAMES = ['conv_kl_ae_v2_7',
         'conv_kl_ae_4',
         'conv_ae_parity_ae_parity_ds_1000']

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


def aggregate():
    for model_name in NAMES:
        path = 'data/grid/results/%s' % model_name
        os.makedirs(path, exist_ok=True)

        data = get_data(model_name)

        print(path, data.shape)

        np.save(path + '/m_5_k_pvalue_3d.npy', data)


def load_aggregated():
    models_data = dict()

    models_data['conv_kl_ae_v2_7'] = np.load(
        'data/grid/results/conv_kl_ae_v2_7/m_5_k_pvalue_3d.npy'
    )

    models_data['conv_kl_ae_4'] = np.load(
        'data/grid/results/conv_kl_ae_4/m_5_k_pvalue_3d.npy'
    )

    models_data['conv_ae_parity_ae_parity_ds_1000'] = np.load(
        'data/grid/results/conv_ae_parity_ae_parity_ds_1000/m_5_k_pvalue_3d.npy'
    )

    return models_data


def transform_x_y_to_grid():
    m5_x = np.arange(1000, 8100, 100)
    k_y = np.arange(0, 3100, 100)

    m5_xx, k_yy = np.meshgrid(m5_x, k_y)

    return m5_xx, k_yy


def transform_z_to_grid(p_values_3d, m5_xx, k_yy):

    z = np.zeros(m5_xx.shape)

    for point in p_values_3d:
        m5 = point[0]
        k = point[1]
        p_value = point[2]

        x = np.where(m5_xx == m5)[1][0]
        y = np.where(k_yy == k)[0][0]

        # print(x, y)

        z[int(y), int(x)] = p_value

    print(z.shape)

    return z


def plot():
    models_data = load_aggregated()

    m5 = models_data['conv_ae_parity_ae_parity_ds_1000'][:, 0]
    k = models_data['conv_ae_parity_ae_parity_ds_1000'][:, 1]
    p_value = models_data['conv_ae_parity_ae_parity_ds_1000'][:, 2]

    p_values_3d = models_data['conv_ae_parity_ae_parity_ds_1000']

    m5_xx, k_yy = transform_x_y_to_grid()

    m5_x = np.arange(1000, 8100, 100)
    k_y = np.arange(0, 3100, 100)

    z = transform_z_to_grid(p_values_3d, m5_xx, k_yy)

    h = plt.contour(m5_x, k_y, z)
    plt.show()


def main():
    pass
    # collect points and aggregate by model
    # aggregate()

    plot()

    # model_path = sys.argv[1].split("@")[4]
    # model_name = model_path.split("/")[-2]

    # grid_path = 'data/grid/%s' % model_name
    # os.makedirs(grid_path, exist_ok=True)

    # data = get_data(model_path, model_type, model_name)
    # data = get_data(model_name)
    # print(model_name, data.shape)
    # plot(data)




if __name__ == "__main__":
    main()

