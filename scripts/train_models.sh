#!/bin/bash

#export path_to_python="/gpfs0/kats/projects/Python-3.8.4/python"
#qsub -l mem_free=50G,h_vmem=50G -pe shared 10 your_script


export path_to_python="/bin/python3"
export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/train_models.py"

MODELS=('conv_ae_2'
        'conv_kl_ae_v2_3'
        'conv_kl_ae_v2_7'
        'conv_kl_ae_4')


for model in "${MODELS[@]}"
do
  echo "Training model: $model"
  export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/train_models.py $model"
  qsub -cwd -q kats.q -S $path_to_python $script_path
done