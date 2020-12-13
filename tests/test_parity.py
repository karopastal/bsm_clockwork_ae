import os
from src.models.conv_ae import ConvAE

# PATH_PARITY_DS = 'data/dataset/12-09-20T17-23-10$5000'
# PATH_V8_DS = 'data/dataset/v8'

""" NORMALIZED """

# PATH_V8_AE_V8_DS_NOR = 'data/models/v8_ae_v8_ds_nor'
# PATH_V8_AE_PARITY_DS_NOR = 'data/models/v8_ae_parity_ds_nor'
# PATH_PARITY_AE_PARITY_DS_1000_NOR = 'data/models/conv_ae/conv_ae_parity_ae_parity_ds_1000/Dec-09-20_T_18-02-46'
# PATH_PARITY_AE_V8_DS_1000_NOR = 'data/models/conv_ae/conv_ae_parity_ae_v8_ds_1000/Dec-10-20_T_12-51-41'

# PATH_PARITY_AE_PARITY_DS_64_NOR = 'data/models/conv_ae/conv_ae_parity_ae_parity_ds_64/Dec-09-20_T_19-09-16'
# PATH_PARITY_AE_V8_DS_64_NOR = '-'   # latest

""" NON NORMALIZED """

# PATH_V8_AE_V8_DS = 'data/models/v8_ae_v8_ds'
# PATH_V8_AE_PARITY_DS = 'data/models/v8_ae_parity_ds'
# PATH_PARITY_AE_PARITY_DS_1000 = 'data/models/conv_ae/conv_ae_parity_ae_parity_ds_1000/Dec-10-20_T_12-32-35'
# PATH_PARITY_AE_V8_DS_1000 = 'data/models/conv_ae/conv_ae_parity_ae_v8_ds_1000/Dec-10-20_T_12-32-35'

# PATH_PARITY_AE_PARITY_DS_64 = 'data/models/conv_ae/conv_ae_parity_ae_parity_ds_64'
# PATH_PARITY_AE_V8_DS_64 = '+'


# def eval_model(model, title=''):
#     # model.plot_progress(title=title)
#     model.eval_model(name=model.name, m_5=6200, k=1000)
#     model.create_loss_distribution(name=model.name, m_5=6200, k=1000)


""" NON NORMALIZED """

# v8_ae_v8_ds = ConvAE(path_model=PATH_V8_AE_V8_DS, path_dataset=PATH_V8_DS)
# eval_model(v8_ae_v8_ds, title="v8_ae_v8_ds")
#
#
# v8_ae_parity_ds = ConvAE(path_model=PATH_V8_AE_PARITY_DS, path_dataset=PATH_PARITY_DS)
# eval_model(v8_ae_parity_ds, title="v8_ae_parity_ds")
#
#
# parity_ae_parity_ds = ConvAE(path_model=PATH_PARITY_AE_PARITY_DS_1000, path_dataset=PATH_PARITY_DS)
# eval_model(parity_ae_parity_ds, title="parity_ae_parity_ds")
#
# parity_ae_v8_ds = ConvAE(path_model=PATH_PARITY_AE_V8_DS_1000, path_dataset=PATH_V8_DS)
# eval_model(parity_ae_v8_ds, title="parity_ae_v8_ds")


""" NORMALIZED """


# v8_ae_v8_ds = ConvAE(path_model=PATH_V8_AE_V8_DS_NOR, path_dataset=PATH_V8_DS)
# eval_model(v8_ae_v8_ds, title="v8_ae_v8_ds_nor")

# v8_ae_parity_ds = ConvAE(path_model=PATH_V8_AE_PARITY_DS_NOR, path_dataset=PATH_PARITY_DS)
# eval_model(v8_ae_parity_ds, title="v8_ae_parity_ds_nor")
#
# parity_ae_parity_ds = ConvAE(path_model=PATH_PARITY_AE_PARITY_DS_1000_NOR, path_dataset=PATH_PARITY_DS)
# eval_model(parity_ae_parity_ds, title="parity_ae_parity_ds_nor")
#
# parity_ae_v8_ds = ConvAE(path_model=PATH_PARITY_AE_V8_DS_1000_NOR, path_dataset=PATH_V8_DS)
# eval_model(parity_ae_v8_ds, title="parity_ae_v8_ds_nor")

""" train: 50,000, test: 25,000 """

PATH_PARITY_DS = ''
PATH_V8_DS = ''


def eval_model(model, title=''):
    # model.plot_progress(title=title)
    model.eval_model(name=model.name, m_5=6200, k=1000)
    model.create_loss_distribution(name=model.name, m_5=6200, k=1000)
