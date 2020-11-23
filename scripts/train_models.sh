#!/bin/bash

#export path_to_python="/gpfs0/kats/projects/Python-3.8.4/python"

export path_to_python="/bin/python3"
export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/build_dataset.py"

MODELS=('conv_ae_1'
        'conv_ae_2'
        'conv_ae_3')

for model in "${MODELS[@]}"
do
  echo "Training model: $model"
  export MODEL_ARG="$model"
  pyenv exec python -m src.models.train_models
#  qsub -cwd -q kats.q -S $path_to_python $script_path
done