#!/bin/bash

#export path_to_python="/gpfs0/kats/projects/Python-3.8.4/python"

export path_to_python="/bin/python3"
export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/train_models.py"

MODEL_PATHS=('data/models/conv_kl_ae/conv_kl_ae_2/Dec-14-20_T_16-20-31'
             'data/models/conv_kl_ae/conv_kl_ae_3/Dec-14-20_T_16-20-31'
             'data/models/conv_kl_ae/conv_kl_ae_4/Dec-14-20_T_16-20-31'
             'data/models/conv_kl_ae/conv_kl_ae_5/Dec-14-20_T_16-20-31'
             'data/models/conv_kl_ae/conv_kl_ae_6/Dec-14-20_T_16-20-31'
             'data/models/conv_kl_ae_v1/conv_kl_ae_v1_1/Dec-14-20_T_17-27-21'
             'data/models/conv_kl_ae_v1/conv_kl_ae_v1_3/Dec-14-20_T_17-27-21'
             'data/models/conv_kl_ae_v1/conv_kl_ae_v1_5/Dec-14-20_T_17-27-21'
             'data/models/conv_kl_ae_v2/conv_kl_ae_v1_1/Dec-14-20_T_17-27-21'
             'data/models/conv_kl_ae_v2/conv_kl_ae_v1_3/Dec-14-20_T_17-27-21'
             'data/models/conv_kl_ae_v2/conv_kl_ae_v1_5/Dec-14-20_T_17-27-21')

for path in "${MODEL_PATHS[@]}"
do
  echo "Training model: $model"
  export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/eval_model.py $path"
  qsub -cwd -q kats.q -S $path_to_python $script_path
done