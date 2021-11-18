from functools import partial
import torch
import seaborn as sns
from hnpe.misc import make_label
from viz import get_posterior
from posterior import build_flow, IdentitySourceLocalization
from simulator import (
    simulator_SourceLocalization, prior_SourceLocalization)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import os


mpl.rcParams['text.usetex'] = True
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.title_fontsize'] = 18
plt.rc('legend', fontsize=14)


def boilerplate(nextra, globalcoord='u', u0=0.50, v0=0.50, sigma=0.00,
                rotation=0.00, round_idx=0):

    # setup the parameters for the example
    meta_parameters = {}
    # how many extra observations to consider
    meta_parameters["n_extra"] = nextra
    # what kind of summary features to use
    meta_parameters["summary"] = 'Identity'
    # the parameters of the ground truth (observed data)
    meta_parameters["theta"] = torch.tensor([0.50, 0.50])
    # which angle to rotate the axis
    meta_parameters["rotation"] = rotation
    # which example case we are considering here
    meta_parameters["case"] = ''.join(
        [f"Microphones_",
         f"nextra_{meta_parameters['n_extra']:02}_",
         f"u_{meta_parameters['theta'][0]:.2f}_",
         f"v_{meta_parameters['theta'][1]:.2f}_",
         f"global_coord_{globalcoord}_",
         f"sigma_{sigma:.2f}"])
    if meta_parameters["rotation"] > 0.00:
        meta_parameters["case"] = meta_parameters["case"] + \
            f"_rotation_{rotation:.2f}"
    # number of rounds to use in the SNPE procedure
    meta_parameters["n_rd"] = 2
    # number of simulations per round
    meta_parameters["n_sr"] = 50_000
    # number of summary features to consider
    meta_parameters["n_sf"] = 1
    # which density estimator to use (maf/mdn/nsf)
    meta_parameters["density"] = 'nsf'
    # how much noise to consider on the output
    meta_parameters["sigma"] = sigma
    # which coordinate to consider as global
    gcoord = {'u': 0, 'v': 1}[globalcoord]
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
                        rotation=rotation)

    # choose how to get the summary features
    summary_net = IdentitySourceLocalization()

    # choose a function which creates a neural network density estimator
    build_nn_posterior = partial(build_flow,
                                 embedding_net=summary_net,
                                 factorized=True,
                                 global_coord=meta_parameters["global_coord"],
                                 density=meta_parameters['density'])

    summary_extractor = None
    posterior = get_posterior(
        simulator, prior, summary_extractor, build_nn_posterior,
        meta_parameters, round_=round_idx
    )

    x0 = simulator(torch.tensor([u0, v0]))
    posterior.set_default_x(x0)

    return posterior


# plot two posterior distributions: one with a single observation and the other
# with N additional observations
def plot_posterior_two_nextra(N, globalcoord, show=False):

    n_samples = 1_000
    u0, v0 = 1.00, 1.00

    samples = {}
    for nextra in [0, N]:
        posterior = boilerplate(nextra, globalcoord=globalcoord, u0=u0, v0=v0)
        samples[nextra] = posterior.sample(
            (n_samples,),
            sample_with_mcmc=False,
            show_progress_bars=False)

    colors = {0: 'C0', N: 'C1'}
    fig, ax = plt.subplots(figsize=(6.5, 6.1))
    df = pd.DataFrame()
    df['x'] = np.concatenate([samples[0][:, 0].numpy(),
                              samples[N][:, 0].numpy()])
    df['y'] = np.concatenate([samples[0][:, 1].numpy(),
                              samples[N][:, 1].numpy()])
    df['N'] = np.array(n_samples*[0] + n_samples*[N])

    sns.kdeplot(data=df, x='x', y='y', hue='N', palette=[colors[0], colors[N]],
                legend=False, levels=5, ax=ax, common_norm=False)

    ax.set_xticks([-2.0, -1.0, 0, 1.0, 2.0])
    ax.set_xticklabels([-2.0, -1.0, 0, 1.0, 2.0], fontsize=18)
    ax.set_yticks([-2.0, -1.0, 0, 1.0, 2.0])
    ax.set_yticklabels([-2.0, -1.0, 0, 1.0, 2.0], fontsize=18)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r'$u$', fontsize=22)
    ax.set_ylabel(r'$v$', fontsize=22)
    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)
    # increase tick width
    ax.tick_params(width=1)
    ax.scatter(u0, v0, marker='*', s=200, c='k', zorder=10)
    ax.scatter(-1, 0, s=150, c='C2')
    ax.scatter(+1, 0, s=150, c='C2')

    fig.savefig(f'figure-posterior-two-nextra-global-coord_{globalcoord}.pdf',
                format='pdf')
    if show:
        fig.show()


# plot the posterior distribution with a single observation
def plot_posterior_single(N=0, globalcoord='u', show=False):

    n_samples = 1_000
    u0, v0 = 1.00, 1.00
    samples = {}
    posterior = boilerplate(N, globalcoord=globalcoord, u0=u0, v0=v0)
    samples[N] = posterior.sample(
        (n_samples,),
        sample_with_mcmc=False,
        show_progress_bars=False)

    fig, ax = plt.subplots(figsize=(6.5, 6.1))
    df = pd.DataFrame()
    df['x'] = samples[N][:, 0].numpy()
    df['y'] = samples[N][:, 1].numpy()

    sns.kdeplot(data=df, x='x', y='y', color='C0',
                legend=False, levels=5, ax=ax)

    ax.set_xticks([-2.0, -1.0, 0, 1.0, 2.0])
    ax.set_xticklabels([-2.0, -1.0, 0, 1.0, 2.0], fontsize=18)
    ax.set_yticks([-2.0, -1.0, 0, 1.0, 2.0])
    ax.set_yticklabels([-2.0, -1.0, 0, 1.0, 2.0], fontsize=18)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r'$u$', fontsize=22)
    ax.set_ylabel(r'$v$', fontsize=22)

    if N > 0:
        ax.set_title(r'$p(u, v | x, \mathcal{X})$', fontsize=22)
    else:
        ax.set_title(r'$p(u, v | x)$', fontsize=22)

    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)
    # increase tick width
    ax.tick_params(width=1)
    ax.scatter(u0, v0, marker='*', s=150, c='k', zorder=10)
    ax.scatter(-1, 0, s=100, c='k')
    ax.scatter(+1, 0, s=100, c='k')
    ax.axvline(x=0, ls='--', c='k', lw=1.0)
    ax.axhline(y=0, ls='--', c='k', lw=1.0)

    fig.savefig(f'figure-posterior-single-global-coord_{globalcoord}.pdf', 
                format='pdf')
    if show:
        fig.show()


# plot the hyperbole showing with the analytic posterior distribution
def plot_hyperbole(show=False):

    def get_itd(theta):
        m1 = np.array([-1, 0])
        m2 = np.array([+1, 0])
        itd = (np.linalg.norm(theta - m1)) - (np.linalg.norm(theta - m2))
        return itd

    u0, v0 = 1.00, 1.00
    theta = torch.tensor([u0, v0])
    x = get_itd(theta)
    varray = np.linspace(0, +2, 100)
    hyperbola = np.sqrt((varray**2)*((x/2)**2)/(1-(x/2)**2)+(x/2)**2)

    fig, ax = plt.subplots(figsize=(6.5, 6.1))
    ax.plot(hyperbola, varray, c='C0', lw=2.0)
    ax.scatter(-1, 0, c='k', s=100)
    ax.scatter(+1, 0, c='k', s=100)
    ax.scatter(u0, v0, marker='*', s=150, c='k', zorder=10)
    ax.axvline(x=0, ls='--', c='k', lw=1.0)
    ax.axhline(y=0, ls='--', c='k', lw=1.0)
    ax.set_xlim(-2, +2)
    ax.set_xticks([-2.0, -1.0, 0, 1.0, 2.0])
    ax.set_xticklabels([-2.0, -1.0, 0, 1.0, 2.0], fontsize=18)
    ax.set_yticks([-2.0, -1.0, 0, 1.0, 2.0])
    ax.set_yticklabels([-2.0, -1.0, 0, 1.0, 2.0], fontsize=18)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r'$u$', fontsize=22)
    ax.set_ylabel(r'$v$', fontsize=22)

    ax.set_title(r'$p(u, v | x)$', fontsize=22)

    fig.savefig('figure-hyperbole.pdf', format='pdf')
    if show:
        fig.show()


plot_hyperbole()
# plot_posterior_single(N=0)
# plot_posterior_single(N=18, globalcoord='u')
# plot_posterior_single(N=18, globalcoord='v')

# LIST_NEXTRA = list([0] + np.unique(np.logspace(0, 3, 20, dtype=int)))[:-2]

# filepath = 'make_figure_distance.pkl'
# if os.path.exists(filepath):
#     distance = joblib.load(filepath)
# else:
#     # set prior distribution for the parameters
#     prior = prior_SourceLocalization(low=torch.tensor([0.0, 0.0]),
#                                      high=torch.tensor([2.0, 2.0]))
#     LIST_THETA = prior.sample((10,))
#     n_samples = 10_000
#     distance = {}
#     for globalcoord in ['u', 'v']:
#         distance[globalcoord] = np.zeros((len(LIST_THETA), len(LIST_NEXTRA)))
#         for i in tqdm(range(len(LIST_THETA))):
#             u0, v0 = LIST_THETA[i]
#             for j, N in enumerate(LIST_NEXTRA):
#                 posterior = boilerplate(N, globalcoord=globalcoord, 
#                                         u0=u0, v0=v0)
#                 samples = posterior.sample(
#                     (n_samples,),
#                     sample_with_mcmc=False,
#                     show_progress_bars=False)
#                 biasN = torch.mean((samples - torch.tensor([u0, v0]))**2,
#                                    dim=0)
#                 varcN = torch.var(samples, dim=0)
#                 distance[globalcoord][i, j] = float(
#                     torch.sum(biasN) + torch.sum(varcN))

#     joblib.dump(distance, 'make_figure_distance.pkl')

# fig, ax = plt.subplots(figsize=(6.5, 6.1))
# for globalcoord in ['u', 'v']:
#     distance_avg = np.mean(distance[globalcoord], axis=0)
#     ax.plot(LIST_NEXTRA, distance_avg, lw=2.0, label=globalcoord)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xticks([1, 10, 100, 1000])
# ax.set_xlim([1, 1000])
# ax.legend()
# fig.show()
