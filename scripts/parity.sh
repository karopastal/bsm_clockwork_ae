#!/bin/bash

#export path_to_python="/gpfs0/kats/projects/Python-3.8.4/python"

export path_to_python="/bin/python3"
export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/train_models.py"

COMMANDS=('src.parity.v8_ae_v8_ds'         # non normalized
          'src.parity.v8_ae_v8_ds_nor'
          'src.parity.v8_ae_parity_ds'     # non normalized
          'src.parity.v8_ae_parity_ds_nor'
          'src.parity.parity_ae_parity_ds' # non normalized
          'src.parity.parity_ae_v8_ds')    # non normalized



#COMMANDS=('src.parity.parity_ae_v8_ds')    # normalized


for command in "${COMMANDS[@]}"
do
  echo "execute: $command"
  export script_path="/gpfs0/kats/users/talpas/projects/bsm_clockwork_ae/scripts/parity.py $command"
  qsub -cwd -q kats.q -S $path_to_python $script_path
done
