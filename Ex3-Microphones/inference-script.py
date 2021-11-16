import os
from collections import OrderedDict
import numpy as np

LIST_NEXTRA = list(np.unique(np.logspace(0, 3, 20, dtype=int)))
for sigma in [0.00, 0.05]:
    for globalcoord in ['u', 'v']:
        for nextra in LIST_NEXTRA:
            parameters = OrderedDict()
            parameters["nextra"] = nextra
            parameters["sigma"] = sigma
            parameters["globalcoord"] = globalcoord
            command = "python inference.py"
            for key in parameters.keys():
                command = command + " --{} {}".format(key, parameters[key])
            os.system(command)
            print(command)
