#!/gpfs0/kats/projects/Python-3.8.4/python

import os
import sys

m_5_k = sys.argv[1]

os.system(
    '/gpfs0/kats/projects/Python-3.8.4/python -m src.grid ' + str(m_5_k)
)
