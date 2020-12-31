import sys
import os
import numpy as np


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


def evaluate_p_value():
    # todo: write p_value of point (m_5, k) => (m_5, k, p_value)
    pass


def create_dataset(m_5, k):
    cwt_generator = CWTGenerator()
    cwt_generator.load(path='data/cwt_generator/default/12-28-20T16-12-13')

    ds = Dataset(cwt_generator,
                 path=TEST_DATASET,
                 logger=str(sys.argv[1]),
                 train_size=1,
                 test_bg_size=1,
                 test_signal_size=25000)

    ds.build_test_signal_dataset(m_5=m_5, k=k)


def save_p_value(model_path, model_type, model_name, m_5, k):

    ae = MODELS[model_type](path_model=model_path,
                            path_dataset=TEST_DATASET)

    p_value = ae.calc_p_value(m_5=m_5, k=k)

    data_path = 'data/grid/%s/m5_%s_k_%s.npy' % (model_name, m_5, k)
    data = np.array([m_5, k, p_value])
    np.save(data_path, data)


def main():
    m_5 = int(sys.argv[1].split("@")[1])
    k = int(sys.argv[1].split("@")[3])
    model_path = sys.argv[1].split("@")[4]

    model_type = model_path.split("/")[-3]
    model_name = model_path.split("/")[-2]

    grid_path = 'data/grid/%s' % model_name
    os.makedirs(grid_path, exist_ok=True)

    save_p_value(model_path, model_type, model_name, m_5, k)


if __name__ == "__main__":
    main()

