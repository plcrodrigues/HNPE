from functools import partial
import torch
import numpy as np
import viz
import joblib
from hnpe.misc import make_label
from posterior import build_flow, IdentityToyModel
from simulator import simulator_ToyModel, prior_ToyModel, get_ground_truth
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.title_fontsize'] = 18 
plt.rc('legend', fontsize=14)
import seaborn as sns
import pandas as pd

def get_posterior(alpha, beta, gamma, nextra, ntrials, noise, naive, round_idx=0):

    # setup the parameters for the example
    meta_parameters = {}
    # how many extra observations to consider
    meta_parameters["n_extra"] = nextra
    # how many trials for each observation
    meta_parameters["n_trials"] = ntrials
    # what kind of summary features to use
    meta_parameters["summary"] = 'Identity'
    # the parameters of the ground truth (observed data)
    meta_parameters["theta"] = torch.tensor([alpha, beta])
    # gamma parameter on x = alpha * beta^gamma + w
    meta_parameters["gamma"] = gamma
    # standard deviation of the noise added to the observations
    meta_parameters["noise"] = noise
    # which example case we are considering here
    meta_parameters["case"] = ''.join([f"Flow/ToyModel_",
                            f"naive_{naive}_", 
                            f"ntrials_{meta_parameters['n_trials']:02}_",
                            f"nextra_{meta_parameters['n_extra']:02}_", 
                            f"alpha_{meta_parameters['theta'][0]:.2f}_",
                            f"beta_{meta_parameters['theta'][1]:.2f}_",
                            f"gamma_{meta_parameters['gamma']:.2f}_",
                            f"noise_{meta_parameters['noise']:.2f}"])

    # number of rounds to use in the SNPE procedure
    meta_parameters["n_rd"] = 5
    # number of simulations per round
    meta_parameters["n_sr"] = 10_000
    # number of summary features to consider
    meta_parameters["n_sf"] = 1
    # label to attach to the SNPE procedure and use for saving files
    meta_parameters["label"] = make_label(meta_parameters)

    # set prior distribution for the parameters
    prior = prior_ToyModel(low=torch.tensor([0.0, 0.0]),
                        high=torch.tensor([1.0, 1.0]))

    # choose how to setup the simulator
    simulator = partial(simulator_ToyModel,
                        n_extra=meta_parameters["n_extra"],
                        n_trials=meta_parameters["n_trials"],
                        p_alpha=prior,
                        gamma=meta_parameters["gamma"],
                        sigma=meta_parameters["noise"])

    # choose the ground truth observation to consider in the inference
    ground_truth = get_ground_truth(meta_parameters, p_alpha=prior)

    # choose how to get the summary features
    summary_net = IdentityToyModel()

    # choose a function which creates a neural network density estimator
    build_nn_posterior = partial(build_flow,
                                 embedding_net=summary_net,
                                 naive=naive)                                

    posterior = viz.get_posterior(
        simulator, 
        prior, 
        build_nn_posterior, 
        meta_parameters, 
        round_=round_idx)

    posterior.set_default_x(ground_truth['observation'])
    return posterior

nsamples = 1_000

samples = {}

N = 250
for nextra in [0, N]:

    naive = False

    posterior = get_posterior(
        alpha=0.50, beta=0.50, gamma=1.00, 
        nextra=nextra, ntrials=1, noise=0.00, naive=naive, round_idx=0)

    samples[nextra] = posterior.sample(
        (nsamples,), 
        sample_with_mcmc=False, 
        show_progress_bars=False)

fig, ax = plt.subplots(figsize=(6.5, 6.1))
df = pd.DataFrame()
df['x'] = np.concatenate([samples[0][:,0].numpy(), samples[N][:,0].numpy()])
df['y'] = np.concatenate([samples[0][:,1].numpy(), samples[N][:,1].numpy()])
df['N'] = np.array(nsamples*[0] + nsamples*[N])
sns.kdeplot(
    data=df, x='x', y='y', hue='N', palette=['C0', 'C1'], 
    legend=False, levels=5, ax=ax)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$\alpha$', fontsize=22)
ax.set_ylabel(r'$\beta$', fontsize=22)
ax.axhline(y=0.50, ls='--', c='k', lw=1.0)
ax.axvline(x=0.50, ls='--', c='k', lw=1.0)
# change all spines
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1)
# increase tick width
ax.tick_params(width=1)
ax.set_title(r'$p(\alpha, \beta| x_0) \quad p(\alpha, \beta| x_0, \mathcal{X})$', fontsize=22)
fig.show()
