import sys
import logging

from src.dataset import Dataset
from src.cwt_generator import CWTGenerator


def init_logger():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename='data/grid.log')

    logging.info('init logging')


# models = dict()


def main():
    m_5 = int(sys.argv[1].split("_")[1])
    k = int(sys.argv[1].split("_")[3])

    cwt_generator = CWTGenerator()
    cwt_generator.load(path='data/cwt_generator/default/12-28-20T16-12-13')

    ds = Dataset(cwt_generator,
                 path='data/dataset/12-28-20T16-12-13$10000',
                 logger=str(sys.argv[1]),
                 train_size=1,
                 test_bg_size=1,
                 test_signal_size=25000)

    ds.build_test_signal_dataset(m_5=m_5, k=k)


if __name__ == "__main__":
    main()

