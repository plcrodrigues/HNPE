from functools import partial

import torch
from hnpe.misc import make_label
from hnpe.inference import run_inference

from posterior import build_flow, IdentityJRNMM
from summary import summary_JRNMM
from viz import get_posterior, display_posterior
from simulator import prior_JRNMM, simulator_JRNMM, get_ground_truth

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Run inference on the JRNMM with no gain parameter'
    )
    parser.add_argument('--C', type=float, default=135.0,
                        help='Ground truth value for C.')
    parser.add_argument('--mu', type=float, default=220.0,
                        help='Ground truth value for mu.')
    parser.add_argument('--sigma', type=float, default=2000.0,
                        help='Ground truth value for sigma.')
    parser.add_argument('--summary', '-s', type=str, default='Fourier',
                        help='Architecture used to compute summary features.')
    parser.add_argument('--viz', action='store_true',
                        help='Show results from previous run.')
    parser.add_argument('--round', '-r', type=int, default=0,
                        help='Show results from previous inference run.')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='How many workers to use.')
    parser.add_argument('--dry', action='store_true',
                        help='Whether to do a dry run.')

    args = parser.parse_args()

    if args.dry:
        maxepochs = 0
        nsr = 10
    else:
        maxepochs = None
        nsr = 50_000

    # setup the parameters for the example
    meta_parameters = {}
    # what kind of summary features to use
    meta_parameters["summary"] = args.summary
    # the parameters of the ground truth (observed data)
    theta = [args.C, args.mu, args.sigma]
    meta_parameters["theta"] = torch.tensor(theta)
    meta_parameters["n_extra"] = 0

    # which example case we are considering here
    meta_parameters["case"] = "JRNMM_nogain_" \
        "C_{:.2f}_" \
        "mu_{:.2f}_" \
        "sigma_{:.2f}".format(meta_parameters["theta"][0],
                              meta_parameters["theta"][1],
                              meta_parameters["theta"][2])

    # number of rounds to use in the SNPE procedure
    meta_parameters["n_rd"] = 2
    # number of simulations per round
    meta_parameters["n_sr"] = nsr
    # number of summary features to consider
    meta_parameters["n_sf"] = 33
    # how many seconds the simulations should have (fs = 128 Hz)
    meta_parameters["t_recording"] = 8
    meta_parameters["n_ss"] = int(128 * meta_parameters["t_recording"])
    # label to attach to the SNPE procedure and use for saving files
    meta_parameters["label"] = make_label(meta_parameters)
    # run example with the chosen parameters
    device = "cpu"

    # set prior distribution for the parameters
    input_parameters = ['C', 'mu', 'sigma']
    prior = prior_JRNMM(parameters=[('C', 10.0, 250.0),
                                    ('mu', 50.0, 500.0),
                                    ('sigma', 100.0, 5000.0)])

    # choose how to setup the simulator
    simulator = partial(simulator_JRNMM,
                        input_parameters=input_parameters,
                        t_recording=meta_parameters["t_recording"],
                        fix_gain=True)

    # choose how to get the summary features
    summary_extractor = summary_JRNMM(
        d_embedding=meta_parameters["n_sf"],
        n_time_samples=meta_parameters["n_ss"],
        type_embedding=meta_parameters["summary"])

    # let's use the log power spectral density instead
    summary_extractor.embedding.net.logscale = True

    # choose the ground truth observation to consider in the inference
    ground_truth = get_ground_truth(meta_parameters,
                                    input_parameters=input_parameters)
    ground_truth['observation'] = summary_extractor(
        ground_truth['observation'])

    # choose a function which creates a neural network density estimator
    build_nn_posterior = partial(build_flow,
                                 embedding_net=IdentityJRNMM(),
                                 naive=True,
                                 z_score_theta=True,
                                 z_score_x=True)

    # decide whether to run inference or viz the results from previous runs
    if not args.viz:
        # run inference procedure over the example
        posteriors = run_inference(
            simulator=simulator,
            prior=prior,
            build_nn_posterior=build_nn_posterior,
            ground_truth=ground_truth,
            meta_parameters=meta_parameters,
            summary_extractor=summary_extractor,
            save_rounds=True, device=device,
            num_workers=args.workers,
            max_num_epochs=maxepochs
        )
    else:
        posterior = get_posterior(
            simulator, prior, summary_extractor, build_nn_posterior,
            meta_parameters, round_=args.round
        )
        display_posterior(posterior, prior, ground_truth)
