#!/bin/bash

#export path_to_python="/gpfs0/kats/projects/Python-3.8.4/python"

export path_to_python="/bin/python3"
export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/train_models.py"

MODELS=('conv_kl_ae_v1_1'
        'conv_kl_ae_v1_2'
        'conv_kl_ae_v1_3'
        'conv_kl_ae_v1_4'
        'conv_kl_ae_v1_5'
        'conv_kl_ae_v1_6'
        'conv_kl_ae_v2_1'
        'conv_kl_ae_v2_2'
        'conv_kl_ae_v2_3'
        'conv_kl_ae_v2_4'
        'conv_kl_ae_v2_5'
        'conv_kl_ae_v2_6')


for model in "${MODELS[@]}"
do
  echo "Training model: $model"
  export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/train_models.py $model"
  qsub -cwd -q kats.q -S $path_to_python $script_path
done