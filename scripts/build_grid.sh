#!/bin/bash

#export path_to_python="/gpfs0/kats/projects/Python-3.8.4/python"

export path_to_python="/bin/python3"
#export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/train_models.py"

K=('1500'
   '2000'
   '2500')

M5=('2000'
    '3000'
    '4000'
    '5000'
    '6000'
    '7000')


#K=('800'
#   '1000')
#
#M5=('6000'
#    '6500'
#    '7000'
#    '7500'
#    '8000')

for k in "${K[@]}"
  do
  for m5 in "${M5[@]}"
    do
      params="m5_${m5}_k_${k}"
      echo "processing: ${params}"
      export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/build_grid.py $params"
      qsub -cwd -q kats.q -S $path_to_python $script_path
    done
#      echo "sleeping: 1 hour"
#      sleep 3600
  done