from functools import partial

import torch

from hnpe.misc import make_label
from hnpe.inference import run_inference

from viz import get_posterior
from viz import display_posterior
from posterior import build_flow, IdentitySourceLocalization
from simulator import (
    simulator_SourceLocalization, prior_SourceLocalization, get_ground_truth)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Run inference on the microphones localization example'
    )
    parser.add_argument('--u', '-u', type=float, default=0.5,
                        help='Ground truth value for u-coordinate.')
    parser.add_argument('--v', '-v', type=float, default=0.5,
                        help='Ground truth value for v-coordinate.')
    parser.add_argument('--sigma', '-s', type=float, default=0.00,
                        help='Level of noise on the observation.')                        
    parser.add_argument('--summary', '-s', type=str, default='Identity',
                        help='Architecture used to compute summary features.')
    parser.add_argument('--viz', action='store_true',
                        help='Show results from previous run.')
    parser.add_argument('--round', '-r', type=int, default=0,
                        help='Show results from previous inference run.')
    parser.add_argument('--dout', type=int, default=1,
                        help='How many features on observed data point.')
    parser.add_argument('--globalcoord', '-g', type=str, default='u',
                        help='Which coordinate to consider as global.')
    parser.add_argument('--nextra', '-n', type=int, default=0,
                        help='How many extra observations to consider.')
    parser.add_argument('--dry', action='store_true',
                        help='Whether to do a dry run.')
    parser.add_argument('--rotation', type=float, default=0.0)
    args = parser.parse_args()

    if args.dry:
        maxepochs = 0
        nsr = 10
        nrd = 1
        save_rounds = False
    else:
        maxepochs = None
        nsr = 50_000
        nrd = 2
        save_rounds = True

    device = "cpu"
    # setup the parameters for the example
    meta_parameters = {}
    # how many extra observations to consider
    meta_parameters["n_extra"] = args.nextra
    # what kind of summary features to use
    meta_parameters["summary"] = args.summary
    # the parameters of the ground truth (observed data)
    meta_parameters["theta"] = torch.tensor([args.u, args.v])
    # which angle to rotate the axis
    meta_parameters["rotation"] = args.rotation
    # which example case we are considering here
    meta_parameters["case"] = ''.join(
        [f"Microphones_",
         f"nextra_{meta_parameters['n_extra']:02}_",
         f"u_{meta_parameters['theta'][0]:.2f}_",
         f"v_{meta_parameters['theta'][1]:.2f}_",
         f"global_coord_{args.globalcoord}"])
    if meta_parameters["rotation"] > 0.00:
        meta_parameters["case"] = meta_parameters["case"] + \
            f"_rotation_{args.rotation:.2f}"
    # number of rounds to use in the SNPE procedure
    meta_parameters["n_rd"] = nrd
    # number of simulations per round
    meta_parameters["n_sr"] = nsr
    # number of summary features to consider
    meta_parameters["n_sf"] = args.dout
    # which density estimator to use (maf/mdn/nsf)
    meta_parameters["density"] = 'maf'
    # how much noise to consider on the output
    meta_parameters["sigma"] = args.sigma
    # which coordinate to consider as global
    gcoord = {'u': 0, 'v': 1}[args.globalcoord]
    meta_parameters["global_coord"] = gcoord
    # label to attach to the SNPE procedure and use for saving files
    meta_parameters["label"] = make_label(meta_parameters)

    # set prior distribution for the parameters
    prior = prior_SourceLocalization(low=torch.tensor([0.0, 0.0]),
                                     high=torch.tensor([2.0, 2.0]))

    # choose how to setup the simulator
    simulator = partial(simulator_SourceLocalization,
                        n_extra=meta_parameters["n_extra"],
                        global_coord=meta_parameters["global_coord"],
                        p_local=prior,
                        dout=meta_parameters["n_sf"],
                        sigma=meta_parameters["sigma"],
                        rotation=args.rotation)

    # choose the ground truth observation to consider in the inference
    ground_truth = get_ground_truth(meta_parameters, p_local=prior)

    # choose how to get the summary features
    summary_net = IdentitySourceLocalization()

    # choose a function which creates a neural network density estimator
    build_nn_posterior = partial(build_flow,
                                 embedding_net=summary_net,
                                 factorized=True,
                                 global_coord=meta_parameters["global_coord"],
                                 density=meta_parameters['density'])

    # decide whether to run inference or viz the results from previous runs
    if not args.viz:
        # run inference procedure over the example
        posteriors = run_inference(
            simulator=simulator,
            prior=prior,
            build_nn_posterior=build_nn_posterior,
            ground_truth=ground_truth,
            meta_parameters=meta_parameters,
            save_rounds=save_rounds,
            device=device,
            max_num_epochs=maxepochs
        )
    else:
        summary_extractor = None
        posterior = get_posterior(
            simulator, prior, summary_extractor, build_nn_posterior,
            meta_parameters, round_=args.round
        )
        display_posterior(posterior, prior, ground_truth)
