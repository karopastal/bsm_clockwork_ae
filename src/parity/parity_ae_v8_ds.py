from src.cwt_generator import CWTGenerator
from src.dataset import Dataset
from src.models.conv_ae import ConvAE

#
# cwt_generator = CWTGenerator()
# cwt_generator.create(name='default')
#
# ds = Dataset(cwt_generator,
#              train_size=5000,
#              test_bg_size=2500,
#              test_signal_size=2500)
#
# ds.build_train_dataset()
# ds.build_test_bg_dataset()
# ds.build_test_signal_dataset(m_5=6200, k=1000)


path_dataset = 'data/dataset/v8'

optimizer = 'adam'

conv_ae = ConvAE(path_dataset=path_dataset,
                 name='conv_ae_parity_1000',
                 optimizer=optimizer)

conv_ae.train_model(epochs=100, batch_size=1000)

conv_ae = ConvAE(path_dataset=path_dataset,
                 name='conv_ae_parity_64',
                 optimizer=optimizer)

conv_ae.train_model(epochs=100, batch_size=64)
