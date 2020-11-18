import numpy as np
from src.cwt_generator import CWTGenerator
from src.dataset import Dataset

cwt_generator = CWTGenerator()
cwt_generator.load(path='data/cwt_generator/type_1/11-16-20T12-36-42')
ds = Dataset(cwt_generator, train_size=10, test_bg_size=5, test_signal_size=5)

ds.build_train_dataset()
ds.build_test_bg_dataset()
ds.build_test_signal_dataset(m_5=7000, k=700)

train = np.load(ds.path_train)
test_background = np.load(ds.path_test_background)
test_signal = np.load(ds.path_test_background)

print(train.shape, test_background.shape, test_signal.shape)

cwt_generator.plot(train[0])
cwt_generator.plot(test_background[0])
cwt_generator.plot(test_signal[0])
