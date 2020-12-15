from src.models.conv_kl_ae import ConvKLAE
from src.models.conv_kl_ae_v1 import ConvKLAEV1
from src.models.conv_kl_ae_v2 import ConvKLAEV2

import sys
import json

TEST_DATASET = 'data/dataset/12-09-20T17-23-10$5000'

MODELS = {
    'conv_kl_ae': ConvKLAE,
    'conv_kl_ae_v1': ConvKLAEV1,
    'conv_kl_ae_v2': ConvKLAEV2
}


def eval_model(model, file_name='', base_dir='', title=''):

    model.plot_progress(title=title,
                        file_name=file_name,
                        base_dir=base_dir)

    model.create_loss_distribution(m_5=6200,
                                   k=1000,
                                   file_name=file_name)


def main():

    model_path = sys.argv[1]
    config_path = model_path + '/config.json'
    config = json.loads(config_path)
    model_type = config['base_dir'].split("/")[-3]
    model_name = config['base_dir'].split("/")[-2]

    title = '%s rho: %s, beta: %s' % (model_name,
                                      config['rho'],
                                      config['beta'])

    ae = MODELS[model_type](path_model=model_path,
                            path_dataset=TEST_DATASET)

    eval_model(ae,
               base_dir=model_path,
               file_name=model_name,
               title=title)


if __name__ == "__main__":
    main()