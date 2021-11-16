from pathlib import Path
import matplotlib.pyplot as plt
from sbi import utils as sbi_utils
from sbi import analysis as sbi_analysis
from sbi.inference.posteriors.direct_posterior import DirectPosterior
import torch


def get_posterior(simulator, prior, summary_extractor, build_nn_posterior,
                  meta_parameters, round_=0):

    folderpath = Path.cwd() / "results" / meta_parameters["label"]

    # load ground truth
    ground_truth = torch.load(folderpath / "ground_truth.pkl",
                              map_location="cpu")

    batch_theta = prior.sample((2,))
    batch_x = simulator(batch_theta)
    if summary_extractor is not None:
        batch_x = summary_extractor(batch_x)

    nn_posterior = build_nn_posterior(batch_theta=batch_theta,
                                      batch_x=batch_x)
    nn_posterior.eval()
    posterior = DirectPosterior(
        method_family="snpe", neural_net=nn_posterior, prior=prior,
        x_shape=ground_truth["observation"].shape
    )

    state_dict_path = folderpath / f"nn_posterior_round_{round_:02}.pkl"
    posterior.net.load_state(state_dict_path)
    posterior = posterior.set_default_x(ground_truth["observation"])

    return posterior


def display_posterior(posterior, prior, ground_truth):

    samples = posterior.sample((10000,), sample_with_mcmc=False)
    xlim = [[prior.support.base_constraint.lower_bound[i], prior.support.base_constraint.upper_bound[i]]
            for i in range(len(prior.support.base_constraint.lower_bound))]
    if 'theta' in ground_truth:
        fig, ax = sbi_analysis.pairplot(
            samples, limits=xlim, points=ground_truth['theta'],
            points_colors='r', points_offdiag={'markersize': 6}
        )
    else:
        fig, ax = sbi_utils.pairplot(samples, limits=xlim)
    plt.show()
