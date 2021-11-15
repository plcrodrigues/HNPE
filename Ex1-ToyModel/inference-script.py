import os
from collections import OrderedDict

LIST_NEXTRA = [0, 1, 5, 10, 25, 50, 100, 250, 500]
naive = False
for nextra in LIST_NEXTRA:
    for noise in [0.00, 0.05]:
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