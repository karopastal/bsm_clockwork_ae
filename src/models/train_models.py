import sys
import logging
from src.models.conv_ae import conv_ae_1, conv_ae_2, conv_ae_3


def init_logger():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename='data/test_batch_training.log')

    logging.info('init logging')


models = dict()

""" conv ae """

models['conv_ae_1'] = conv_ae_1
models['conv_ae_2'] = conv_ae_2
models['conv_ae_3'] = conv_ae_3


def main():
    init_logger()

    model = sys.argv[1]
    logging.info(model)


if __name__ == "__main__":
    main()
