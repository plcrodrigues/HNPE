import os
from collections import OrderedDict
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--globalcoord', '-g', type=str,
                    help='Which coordinate to consider as global.')
args = parser.parse_args()

LIST_NEXTRA = [0] + list(np.unique(np.logspace(0, 3, 20, dtype=int)))
sigma = 0.00
for nextra in LIST_NEXTRA:
    parameters = OrderedDict()
    parameters["nextra"] = nextra
    parameters["sigma"] = sigma
    parameters["globalcoord"] = args.globalcoord
    command = "python inference.py"
    for key in parameters.keys():
        command = command + " --{} {}".format(key, parameters[key])
    os.system(command)
    print(command)
