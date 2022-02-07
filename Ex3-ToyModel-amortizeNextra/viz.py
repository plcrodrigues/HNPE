from cProfile import label
from pathlib import Path

import torch
import numpy as np
from sbi import utils as sbi_utils
from sbi.inference.posteriors.direct_posterior import DirectPosterior

import matplotlib.pyplot as plt

from sbi.utils.sbiutils import standardizing_net


def get_posterior(simulator, prior, build_nn_posterior, meta_parameters,
                  round_=0, build_aggregate_before=None, ground_truth=None): 

    folderpath = Path.cwd() / "results" / meta_parameters["label"]
    print(folderpath)
    
    if ground_truth is None:
        # load ground truth
        ground_truth = torch.load(folderpath / "ground_truth.pkl",
                                map_location="cpu")

    x_obs = ground_truth["observation"].clone()

    if (x_obs[0].shape[0] == 3) and meta_parameters["norm_before"]:
        print('norm_before')
        stand = standardizing_net(x_obs[0][:2])
        path = folderpath / f"stand_net_round_{round_:02}.pkl"
        stand.load_state_dict(torch.load(path))
        x0_n = torch.cat([x_obs[0][0].reshape(-1,1), x_obs[0][2].reshape(-1,1)], dim=1)
        x0_n = stand(x0_n)

        x_obs[0][0] = x0_n[0][0]
        x_obs[0][2] = x0_n[0][1]

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
    posterior = posterior.set_default_x(x_obs)

    return posterior


def display_posterior(posterior, prior):

    n_samples = 100_000

    samples = posterior.sample((n_samples,), sample_with=None)
    print(samples.shape)

    xlim = [[prior.support.base_constraint.lower_bound[i],
             prior.support.base_constraint.upper_bound[i]]
            for i in range(2)]
    fig, axes = sbi_utils.pairplot(samples, limits=xlim)

    return fig, axes

if __name__ == "__main__":
    from torch.distributions import Categorical
    from simulator import prior_ToyModel, get_ground_truth

    meta_parameters = {}
    meta_parameters["theta"] = torch.tensor([0.3, 0.7])
    meta_parameters["gt_nextra"] = 10
    meta_parameters["gamma"] = 1.
    meta_parameters["noise"] = 0.
    meta_parameters["nextra_range"] = 40
    meta_parameters["aggregate_method"] = 'mean'

    nextra_probs = torch.ones(meta_parameters["nextra_range"])/meta_parameters["nextra_range"]

    gt_nextra_prob = torch.zeros_like(nextra_probs)
    gt_nextra_prob[meta_parameters['gt_nextra']] = 1
    p_gt_nextra = Categorical(gt_nextra_prob)

    prior = prior_ToyModel(low=torch.tensor([0.0, 0.0]),
                           high=torch.tensor([1.0, 1.0]))

    meta_parameters["n_trials"] = 1
    ground_truth = get_ground_truth(meta_parameters, prior, p_gt_nextra)
    
    x_obs = ground_truth["observation"][0].numpy()
    print(x_obs)
    xn = ground_truth["extra_obs"].numpy()
    print(xn)
    print(int(x_obs[1]))
    print(np.concatenate((x_obs[0].reshape(-1,1), xn)))

    stand = standardizing_net(x_obs)
    print(stand.state_dict())
    print(stand(x_obs[:2]))

