from src.models.conv_ae import ConvAE
from src.models.conv_kl_ae import ConvKLAE
from src.models.conv_kl_ae_v1 import ConvKLAEV1
from src.models.conv_kl_ae_v2 import ConvKLAEV2

import sys
import json

TEST_DATASET = 'data/dataset/12-28-20T16-12-13$10000'

MODELS = {
    'conv_ae': ConvAE,
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
                                   base_dir=base_dir,
                                   file_name=file_name)


def main():

    model_path = sys.argv[1]

    model_type = model_path.split("/")[-3]
    model_name = model_path.split("/")[-2]

    if model_type == 'conv_ae':
        rho = 0
        beta = 0
    else:
        config_path = model_path + '/config.json'

        with open(config_path) as f:
            config = json.load(f)

        rho = config['rho']
        beta = config['beta']

    title = '%s rho: %s, beta: %s' % (model_name, rho, beta)

    ae = MODELS[model_type](path_model=model_path,
                            path_dataset=TEST_DATASET)

    eval_model(ae,
               base_dir=model_path,
               file_name=model_name,
               title=title)


if __name__ == "__main__":
    main()
