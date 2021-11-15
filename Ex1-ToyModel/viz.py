from pathlib import Path

import torch
from sbi import utils as sbi_utils
from sbi.inference.posteriors.direct_posterior import DirectPosterior


def get_posterior(simulator, prior, build_nn_posterior, meta_parameters,
                  round_=0):

    folderpath = Path.cwd() / "results" / meta_parameters["label"]
    print(folderpath)

    # load ground truth
    ground_truth = torch.load(folderpath / "ground_truth.pkl",
                              map_location="cpu")

    # Construct posterior
    batch_theta = prior.sample((2,))
    batch_x = simulator(batch_theta)
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

    n_samples = 100_000

    samples = posterior.sample((n_samples,), sample_with_mcmc=False)

    xlim = [[prior.support.base_constraint.lower_bound[i],
             prior.support.base_constraint.upper_bound[i]]
            for i in range(2)]
    fig, axes = sbi_utils.pairplot(samples, limits=xlim)

    return fig, axes
