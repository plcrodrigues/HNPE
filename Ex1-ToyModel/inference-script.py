import os
from collections import OrderedDict
import numpy as np


LIST_NEXTRA = list(np.unique(np.logspace(0, 3, 20, dtype=int)))

naive = False
noise = 0.05
for nextra in LIST_NEXTRA:
    parameters = OrderedDict()
    parameters["nextra"] = nextra
    parameters["noise"] = noise
    command = "python inference.py"
    for key in parameters.keys():
        command = command + " --{} {}".format(key, parameters[key])
    if naive:
        command = command + " --naive"
    os.system(command)
    print(command)