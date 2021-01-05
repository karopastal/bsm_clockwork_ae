#!/bin/bash

#export path_to_python="/gpfs0/kats/projects/Python-3.8.4/python"

export path_to_python="/bin/python3"
#export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/train_models.py"

#'data/models/conv_ae/conv_ae_2/Dec-28-20_T_17-27-12'
#'data/models/conv_kl_ae/conv_kl_ae_4/Dec-28-20_T_20-08-43'
#'data/models/conv_kl_ae_v2/conv_kl_ae_v2_7/Dec-28-20_T_20-08-42'

#MODEL_PATHS=('data/models/conv_ae/conv_ae_parity_ae_parity_ds_1000/Dec-09-20_T_18-02-46')

MODEL_PATH="data/models/conv_kl_ae/conv_kl_ae_4/Dec-28-20_T_20-08-43"


K=('1500'
   '2000'
   '2500')

M5=('2000'
    '3000'
    '4000'
    '5000'
    '6000'
    '7000')

for k in "${K[@]}"
  do
  for m5 in "${M5[@]}"
    do
      params="m5@${m5}@k@${k}@${MODEL_PATH}"
      echo "processing: ${params}"
      export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/build_grid.py $params"
      qsub -cwd -q kats.q -S $path_to_python $script_path
    done
#      echo "sleeping: 1 hour"
#      sleep 3600
  done


K=('200'
   '400'
   '600'
   '800'
   '1000')

M5=('6000'
    '6500'
    '7000'
    '7500'
    '8000')

for k in "${K[@]}"
  do
  for m5 in "${M5[@]}"
    do
      params="m5@${m5}@k@${k}@${MODEL_PATH}"
      echo "processing: ${params}"
      export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/build_grid.py $params"
      qsub -cwd -q kats.q -S $path_to_python $script_path
    done
#      echo "sleeping: 1 hour"
#      sleep 3600
  done