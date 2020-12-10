from src.cwt_generator import CWTGenerator
from src.dataset import Dataset
from src.models.conv_ae import ConvAE

path_dataset = 'data/dataset/v8'

optimizer = 'adam'

conv_ae = ConvAE(path_dataset=path_dataset,
                 name='conv_ae_parity_ae_v8_ds_1000',
                 optimizer=optimizer)

conv_ae.train_model(epochs=100, batch_size=1000)

conv_ae = ConvAE(path_dataset=path_dataset,
                 name='conv_ae_parity_ae_v8_ds_64',
                 optimizer=optimizer)

conv_ae.train_model(epochs=100, batch_size=64)
