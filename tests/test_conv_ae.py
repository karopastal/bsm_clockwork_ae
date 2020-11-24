from src.models.conv_ae import ConvAE

PATH_DATASET = 'data/dataset/11-18-20T23-18-18$25000'


def eval_model(model, title=''):
    # model.plot_progress(title=title)
    model.eval_model(name=model.name, m_5=6500, k=1000)
    model.create_loss_distribution(name=model.name, m_5=6500, k=1000)


models = list()

# path_conv_ae_1 = 'data/models/conv_ae/conv_ae_1/Nov-23-20_T_14-49-13'
# conv_ae_1 = ConvAE(path_model=path_conv_ae_1,
#                    path_dataset=PATH_DATASET)
# models.append(conv_ae_1)

path_conv_ae_2 = 'data/models/conv_ae/conv_ae_2/Nov-23-20_T_14-49-13'
conv_ae_2 = ConvAE(path_model=path_conv_ae_2,
                   path_dataset=PATH_DATASET)

models.append(conv_ae_2)

path_conv_ae_3 = 'data/models/conv_ae/conv_ae_3/Nov-23-20_T_14-49-14'
conv_ae_3 = ConvAE(path_model=path_conv_ae_3,
                   path_dataset=PATH_DATASET)

models.append(conv_ae_3)

# path_conv_ae_4 = 'data/models/conv_ae/conv_ae_4/Nov-23-20_T_14-50-58'
# conv_ae_4 = ConvAE(path_model=path_conv_ae_4,
#                    path_dataset=PATH_DATASET)
#
# models.append(conv_ae_4)

for i, model in enumerate(models):
    eval_model(model, title="conv_ae_%s" % (i+1,))


