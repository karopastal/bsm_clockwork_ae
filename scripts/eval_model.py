#!/gpfs0/kats/projects/Python-3.8.4/python

import os
import sys

model_path = sys.argv[1]

os.system(
    '/gpfs0/kats/projects/Python-3.8.4/python -m tests.evaluate_model ' + str(model_path)
)
