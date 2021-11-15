import os
from collections import OrderedDict

# LIST_THETA = [[135.0, 220.0, 2000.0, 0.0],
#               [135.0, 220.0, 2000.0, -10.0],
#               [135.0, 220.0, 2000.0, 10.0],
#               [270.0, 220.0, 2000.0, 0.0],
#               [68.0, 220.0, 2000.0, 0.0],
#               [135.0, 110.0, 2000.0, 0.0],
#               [68.0, 220.0, 2000.0, 10.0],
#               [270.0, 220.0, 2000.0, 10.0],
#               [135.0, 110.0, 1000.0, 0.0],
#               [135.0, 220.0, 1000.0, 0.0]
#               ] 

LIST_NEXTRA = [20]

LIST_CONFIGS = [(True, [135.0, 220.0, 1000.0, 0.0], False),
                (True, [135.0, 220.0, 1000.0, 0.0], True),
                (False, [135.0, 220.0, 1000.0, 0.0], True),                
                ]

if __name__ == "__main__":

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--naive', action='store_true',
    #                     help='Use the naive posterior or not.')                        
    # args = parser.parse_args()
    # naive = args.naive

    # # whether to aggregate the extra observations
    # aggregate = True

    # for theta in LIST_THETA:
    #     for nextra in LIST_NEXTRA:
    #         parameters = OrderedDict()
    #         parameters["summary"] = 'Fourier'
    #         parameters["nextra"] = nextra
    #         parameters["workers"] = 40
    #         parameters["C"] = theta[0]
    #         parameters["mu"] = theta[1]
    #         parameters["sigma"] = theta[2]
    #         parameters["gain"] = theta[3]
    #         case = f"JRNMM_nextra_{nextra}"
    #         for key in parameters.keys():
    #             case = "_".join([case, key, str(parameters[key])])
    #         command = "python inference.py"
    #         for key in parameters.keys():
    #             command = command + " --{} {}".format(key, parameters[key])
    #         if naive:
    #             command = command + " --naive"
    #         if aggregate:
    #             command = command + " --aggregate"
    #         os.system(command)
    #         print(command)

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--naive', action='store_true',
    #                     help='Use the naive posterior or not.')
    # args = parser.parse_args()
    # naive_input = args.naive

    for config in LIST_CONFIGS:        

        naive = config[0]
        theta = config[1]
        aggregate = config[2]

        for nextra in LIST_NEXTRA:

            parameters = OrderedDict()
            parameters["summary"] = 'Fourier'
            parameters["nextra"] = nextra
            parameters["workers"] = 40
            parameters["C"] = theta[0]
            parameters["mu"] = theta[1]
            parameters["sigma"] = theta[2]
            parameters["gain"] = theta[3]
            case = f"JRNMM_nextra_{nextra}"
            for key in parameters.keys():
                case = "_".join([case, key, str(parameters[key])])
            command = "python inference.py"
            for key in parameters.keys():
                command = command + " --{} {}".format(key, parameters[key])
            if naive:
                command = command + " --naive"
            if aggregate:
                command = command + " --aggregate"
            os.system(command)
            print(command)