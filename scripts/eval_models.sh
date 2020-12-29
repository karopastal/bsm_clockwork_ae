#!/bin/bash

#export path_to_python="/gpfs0/kats/projects/Python-3.8.4/python"

export path_to_python="/bin/python3"
#export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/train_models.py"

#'data/models/conv_ae/conv_ae_2/Dec-28-20_T_17-27-12'
#'data/models/conv_kl_ae/conv_kl_ae_4/Dec-28-20_T_20-08-43'
#'data/models/conv_kl_ae_v2/conv_kl_ae_v2_3/Dec-28-20_T_17-27-13'
#'data/models/conv_kl_ae_v2/conv_kl_ae_v2_7/Dec-28-20_T_20-08-42'

MODEL_PATHS=('data/models/conv_ae/conv_ae_parity_ae_parity_ds_1000/Dec-09-20_T_18-02-46')

for path in "${MODEL_PATHS[@]}"
do
  echo "Training model: $model"
  export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/eval_model.py $path"
  qsub -cwd -q kats.q -S $path_to_python $script_path
done