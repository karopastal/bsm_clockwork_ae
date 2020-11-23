#!/gpfs0/kats/projects/Python-3.8.4/python

import os
import sys

model = sys.argv[1]

os.system('/gpfs0/kats/projects/Python-3.8.4/python -m src.models.train_models ' + str(model))
