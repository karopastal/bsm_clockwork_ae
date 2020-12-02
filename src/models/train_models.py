import sys
import logging

from src.models.conv_ae import conv_ae_1,\
                               conv_ae_2,\
                               conv_ae_3,\
                               conv_ae_4


def init_logger():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename='data/batch_training.log')

    logging.info('init logging')


models = dict()

""" conv ae """

# models['conv_ae_1'] = conv_ae_1
# models['conv_ae_2'] = conv_ae_2
# models['conv_ae_3'] = conv_ae_3
# models['conv_ae_4'] = conv_ae_4


def job_status():
    # todo: read the status from 'training.log'
    #       of each model and print it to central file
    #       => do it as make training_status shell script
    pass


def main():
    init_logger()

    model_arg = sys.argv[1]
    models[model_arg]()
    # logging.info()


if __name__ == "__main__":
    main()
