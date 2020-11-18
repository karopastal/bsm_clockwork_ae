#!/bin/python3 -u

from src.cwt_generator import CWTGenerator
from src.dataset import Dataset

cwt_generator = CWTGenerator()
cwt_generator.create(name='default')

ds = Dataset(cwt_generator,
             train_size=35000,
             test_bg_size=10000,
             test_signal_size=10000)

ds.build_train_dataset()
ds.build_test_bg_dataset()
ds.build_test_signal_dataset(m_5=6000, k=1000)
ds.build_test_signal_dataset(m_5=6500, k=1000)
ds.build_test_signal_dataset(m_5=7000, k=1000)
ds.build_test_signal_dataset(m_5=7500, k=1000)
