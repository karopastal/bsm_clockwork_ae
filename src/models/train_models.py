import sys
import logging

from src.models.conv_ae import conv_ae_2
from src.models.conv_kl_ae import conv_kl_ae_4
from src.models.conv_kl_ae_v2 import conv_kl_ae_v2_3, conv_kl_ae_v2_7


def init_logger():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename='data/batch_training.log')

    logging.info('init logging')


models = dict()

models['conv_ae_2'] = conv_ae_2
models['conv_kl_ae_4'] = conv_kl_ae_4
models['conv_kl_ae_v2_3'] = conv_kl_ae_v2_3
models['conv_kl_ae_v2_7'] = conv_kl_ae_v2_7


def main():
    init_logger()

    model_arg = sys.argv[1]
    models[model_arg]()


if __name__ == "__main__":
    main()
