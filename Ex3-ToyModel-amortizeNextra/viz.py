from pathlib import Path

import torch
from sbi import utils as sbi_utils
from sbi.inference.posteriors.direct_posterior import DirectPosterior

from functools import partial


def get_posterior(simulator, prior, build_nn_posterior, meta_parameters,
                  round_=0, build_aggregate_before=None, ground_truth=None): 

    folderpath = Path.cwd() / "results" / meta_parameters["label"]
    print(folderpath)
    
    if ground_truth is None:
        # load ground truth
        ground_truth = torch.load(folderpath / "ground_truth.pkl",
                                map_location="cpu")

    ## --------------------------- added ------------------------------- ##
    if build_aggregate_before is not None:
        aggregate_before = build_aggregate_before(x_ref=torch.zeros((
            meta_parameters["n_sr"], 
            meta_parameters["n_trials"], 
            meta_parameters["n_extra"])
            ))
        # load parameters (mean, std) defined based on training simulations
        path = folderpath / f"norm_agg_before_net_round_{round_:02}.pkl"
        aggregate_before.load_state_dict(torch.load(path))
        # normalize and aggregate the groundtruth 
        ground_truth["observation"] = aggregate_before(ground_truth["observation"])
    else:
        aggregate_before = None
    ## ----------------------------------------------------------------- ##
        
    # Construct posterior
    batch_theta = prior.sample((2,))
    batch_x = simulator(batch_theta)
    ## ---------- added ----------- ##
    # normalize and aggregate simulations 
    if aggregate_before is not None:
        batch_x = aggregate_before(batch_x)
    ## ---------------------------- ##

    nn_posterior = build_nn_posterior(batch_theta=batch_theta,
                                      batch_x=batch_x)
    nn_posterior.eval()
    posterior = DirectPosterior(
        method_family="snpe",
        neural_net=nn_posterior,
        prior=prior,
        x_shape=ground_truth["observation"].shape
    )

    # Load learned posterior
    state_dict_path = folderpath / f"nn_posterior_round_{round_:02}.pkl"
    posterior.net.load_state(state_dict_path)

    # Set the default conditioning as the observation
    posterior = posterior.set_default_x(ground_truth["observation"])

    return posterior


def display_posterior(posterior, prior):

    n_samples = 1

    samples = posterior.sample((n_samples,), sample_with=None)

    xlim = [[prior.support.base_constraint.lower_bound[i],
             prior.support.base_constraint.upper_bound[i]]
            for i in range(2)]
    fig, axes = sbi_utils.pairplot(samples, limits=xlim)

    return fig, axes
