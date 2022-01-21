from curses import meta
from functools import partial

import torch
import matplotlib.pyplot as plt

from hnpe.misc import make_label
from hnpe.inference import run_inference

from viz import get_posterior
from viz import display_posterior, display_analytic_posterior, plot_2d_pdf_contours
from posterior import build_flow, IdentityToyModel
from simulator import simulator_ToyModel_amortizeNextra, prior_ToyModel, get_ground_truth, preprocess_for_amortizeNextra

from torch.distributions import Categorical

"""
In this example, we consider the ToyModel setting in which the simulator has
two input parameters [alpha, beta] and generates x = alpha * beta^gamma + eps,
where gamma is a fixed known parameter of the simulator, and eps is a Gaussian
white noise with standard deviation sigma. Because the observation is a product
of two parameters, we may expect an indeterminacy when trying to estimate them
from a given observation xo. To try and break this, we consider that each x0 is
accompanied by a few other observations x1, ..., xN which all share the same
parameter beta but with different values for alpha. Our goal then is to use
this extra information to obtain the posterior distribution of
p(alpha, beta | x0, x1, ..., xN)
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Run inference on the Toy Model'
    )
    parser.add_argument('--alpha', '-a', type=float, default=0.5,
                        help='Ground truth value for alpha.')
    parser.add_argument('--beta', '-b', type=float, default=0.5,
                        help='Ground truth value for beta.')
    parser.add_argument('--gamma', '-y', type=float, default=1.0,
                        help='Ground truth value for gamma.')
     ## ----------------------- added ----------------------------- ##
    parser.add_argument('--gt_nextra', type=int, default=0,
                        help='Ground truth value for nextra.')  
     ## ----------------------------------------------------------- ##               
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Standard deviation of the Gaussian noise.')
    parser.add_argument('--summary', '-s', type=str, default='Identity',
                        help='Architecture used to compute summary features.')
    parser.add_argument('--viz', action='store_true',
                        help='Show results from previous run.')
    parser.add_argument('--naive', action='store_true',
                        help='Use naive posterior estimation.')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate the extra observations.')
    parser.add_argument('--round', '-r', type=int, default=0,
                        help='Show results from previous inference run.')
    ## ----------------------- added ----------------------------- ##
    parser.add_argument('--nextra_range', '-n', type=int, default=40,
                        help='The range of values of nextra (number of extra observations).')
    ## ----------------------------------------------------------- ##
    parser.add_argument('--ntrials', '-t', type=int, default=1,
                        help='How many trials to consider.')
    parser.add_argument('--dry', action='store_true')

    parser.add_argument('--aggregate_before', action='store_true', 
                        help='Aggregate the extra observations before training')
    parser.add_argument('--norm_before', action='store_true', 
                        help='Normalize data and groundtruth before training')
    parser.add_argument('--aggregate_method', type=str, default='mean', choices=[None, 'mean'],
                        help='Aggregation method for extra observations.')              

    args = parser.parse_args()

    assert args.nextra_range !=0 , 'To amortize on nextra, the range of values cannot be zero or 1. It has to be at least 2.'

    if args.dry:
        # the dryrun serves just to check if all is well
        nrd = 1
        nsr = 10
        maxepochs = 0
        saverounds = False
    else:
        ## ------------ changed ----------------- ##
        nrd = 1 ## we use NPE not SNPE for amortization
        ## -------------------------------------- ##
        nsr = 10_000
        maxepochs = None
        saverounds = True

    # setup the parameters for the example
    meta_parameters = {}
    # whether to do naive implementation
    meta_parameters["naive"] = args.naive
    ## ----------------------- added ----------------------------- ##
    # the range of values of how many extra observations to consider
    meta_parameters["n_extra_range"] = args.nextra_range
    #aggregation method for extra observations 
    meta_parameters["aggregate_method"] = args.aggregate_method
    ## ---------------------------------------------------------- ##
    # how many trials for each observation
    meta_parameters["n_trials"] = args.ntrials
    # what kind of summary features to use
    meta_parameters["summary"] = args.summary
    # the parameters of the ground truth (observed data)
    meta_parameters["theta"] = torch.tensor([args.alpha, args.beta])
    # nextra value of the ground truth
    meta_parameters["gt_nextra"] = args.gt_nextra
    # gamma parameter on x = alpha * beta^gamma + w
    meta_parameters["gamma"] = args.gamma
    # standard deviation of the noise added to the observations
    meta_parameters["noise"] = args.noise
    # aggregate before or not 
    meta_parameters["agg_before"] = args.aggregate_before
    # whether to normalize x0 and xn before training intead of 
    # zscore on x0,nextra,xn during training 
    meta_parameters["norm_before"] = args.norm_before
    # which example case we are considering here

    if meta_parameters["norm_before"]:
        meta_parameters["case"] = "ToyModel_nextra_range_{:02}_" \
                        "naive_{}_aggregate_norm_before_no_zscore_x_{}".format(meta_parameters["n_extra_range"],
                            meta_parameters["naive"], meta_parameters["aggregate_method"], meta_parameters["norm_before"])
    else:
        meta_parameters["case"] = "ToyModel_nextra_range_{:02}_" \
                        "naive_{}_aggregate_{}".format(meta_parameters["n_extra_range"],
                            meta_parameters["naive"], meta_parameters["aggregate_method"])
    
    # number of rounds to use in the SNPE procedure
    meta_parameters["n_rd"] = nrd
    # number of simulations per round
    meta_parameters["n_sr"] = nsr
    # number of summary features to consider
    meta_parameters["n_sf"] = 1
    # label to attach to the SNPE procedure and use for saving files
    meta_parameters["label"] = make_label(meta_parameters)

    ## -------------------- added --------------------- ##
    # set probabilities for categorical distribution on nextra values 
    nextra_probs = torch.ones(meta_parameters['n_extra_range'])*1/meta_parameters['n_extra_range']
    prior_nextra = Categorical(nextra_probs)
    
    # same for ground truth nextra value 
    gt_nextra_prob = torch.zeros_like(nextra_probs)
    gt_nextra_prob[meta_parameters['gt_nextra']] = 1
    p_gt_nextra = Categorical(gt_nextra_prob)
    ## ----------------------------------------------- ##

    # set prior distribution for the parameters
    prior_theta = prior_ToyModel(low=torch.tensor([0.0, 0.0]),
                           high=torch.tensor([1.0, 1.0]))

    # choose how to setup the simulator
    simulator = partial(simulator_ToyModel_amortizeNextra,
                        n_trials=meta_parameters["n_trials"],
                        p_alpha=prior_theta,
                        p_nextra=prior_nextra,
                        gamma=meta_parameters["gamma"],
                        sigma=meta_parameters["noise"],
                        aggregate_method=meta_parameters["aggregate_method"])

    # choose the ground truth observation to consider in the inference
    ground_truth = get_ground_truth(meta_parameters, p_alpha=prior_theta, p_nextra=p_gt_nextra)

    # choose how to get the summary stats
    summary_net = IdentityToyModel()

    # choose a function which creates a neural network density estimator
    build_nn_posterior = partial(build_flow,
                                 embedding_net=summary_net,
                                 naive=args.naive, ## for now always True
                                 z_score_x=not meta_parameters["norm_before"]) 

    # decide whether to run inference or viz the results from previous runs
    if not args.viz:
        # run inference procedure over the example
        posteriors = run_inference(simulator=simulator,
                                   prior=prior_theta,
                                   build_nn_posterior=build_nn_posterior,
                                   ground_truth=None,
                                   meta_parameters=meta_parameters,
                                   summary_extractor=summary_net,
                                   save_rounds=saverounds,
                                   device='cpu',
                                   max_num_epochs=maxepochs)

    else:
        posterior = get_posterior(
            simulator, prior_theta, build_nn_posterior,
            meta_parameters, round_=args.round, ground_truth=ground_truth
        )

        fig_1, ax, x_learned = display_posterior(posterior, prior_theta)
        # plt.savefig(f'pairplot_round{args.round}_nextra_range_{args.nextra_range}_gt_alpha_{args.alpha}_gt_beta_{args.beta}_gt_nextra_{args.gt_nextra}.png')
        # plt.close(fig_1)

        fig_2, x_true, x_plot, y_plot = display_analytic_posterior(prior_theta, ground_truth)
        # plt.savefig(f'analytic_pairplot_gt_nextra_{meta_parameters["gt_nextra"]}.png')

        fig, ax = plot_2d_pdf_contours(x_true, x_learned, x_plot, y_plot)
        plot_title = f'truevslearned_gt_nextra_{meta_parameters["gt_nextra"]}_aggregate_{meta_parameters["aggregate_method"]}.png'
        if meta_parameters["norm_before"]:
            plot_title = f'truevslearned_gt_nextra_{meta_parameters["gt_nextra"]}_aggregate_{meta_parameters["aggregate_method"]}_norm_before_{meta_parameters["norm_before"]}.png'

        plt.savefig(plot_title)