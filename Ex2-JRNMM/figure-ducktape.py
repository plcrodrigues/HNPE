
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sbi_workforce.misc import make_label
from posterior import build_flow, IdentityJRNMM
from summary import summary_JRNMM
from viz import get_posterior
from simulator import prior_JRNMM, simulator_JRNMM
from functools import partial
import numpy as np

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

LIST_NEXTRA = [0, 10, 20, 30, 40]

def boilerplate(theta, nextra, naive, aggregate=True, round_idx=0):

    # setup the parameters for the example
    meta_parameters = {}
    # how many extra observations to consider
    meta_parameters["n_extra"] = nextra
    # what kind of summary features to use
    meta_parameters["summary"] = 'Fourier'
    # the parameters of the ground truth (observed data)
    meta_parameters["theta"] = theta

    # whether to do naive implementation
    meta_parameters["naive"] = naive

    # which example case we are considering here
    meta_parameters["case"] = "JRNMM_nextra_{:02}_" \
        "naive_{}_" \
        "C_{:.2f}_" \
        "mu_{:.2f}_" \
        "sigma_{:.2f}_" \
        "gain_{:.2f}".format(meta_parameters["n_extra"],
                             meta_parameters["naive"],
                             meta_parameters["theta"][0],
                             meta_parameters["theta"][1],
                             meta_parameters["theta"][2],
                             meta_parameters["theta"][3])

    if not aggregate and naive:
        meta_parameters["case"] = meta_parameters["case"] + "_aggregate_False"

    # number of rounds to use in the SNPE procedure
    meta_parameters["n_rd"] = 2
    # number of simulations per round
    meta_parameters["n_sr"] = 50_000
    # number of summary features to consider
    meta_parameters["n_sf"] = 33
    # how many seconds the simulations should have (fs = 128 Hz)
    meta_parameters["t_recording"] = 8
    meta_parameters["n_ss"] = int(128 * meta_parameters["t_recording"])
    # label to attach to the SNPE procedure and use for saving files
    meta_parameters["label"] = make_label(meta_parameters)

    # set prior distribution for the parameters
    input_parameters = ['C', 'mu', 'sigma', 'gain']
    prior = prior_JRNMM(parameters=[('C', 10.0, 250.0),
                                    ('mu', 50.0, 500.0),
                                    ('sigma', 100.0, 5000.0),
                                    ('gain', -20.0, +20.0)])

    # choose how to setup the simulator
    simulator = partial(simulator_JRNMM,
                        input_parameters=input_parameters,
                        t_recording=meta_parameters["t_recording"],
                        n_extra=meta_parameters["n_extra"],
                        p_gain=prior)

    # choose how to get the summary features
    summary_extractor = summary_JRNMM(n_extra=meta_parameters["n_extra"],
                                      d_embedding=meta_parameters["n_sf"],
                                      n_time_samples=meta_parameters["n_ss"],
                                      type_embedding=meta_parameters["summary"])

    # let's use the log power spectral density instead
    summary_extractor.embedding.net.logscale = True

    # choose a function which creates a neural network density estimator
    build_nn_posterior = partial(build_flow,
                                 embedding_net=IdentityJRNMM(),
                                 naive=meta_parameters["naive"],
                                 aggregate=aggregate,
                                 z_score_theta=True,
                                 z_score_x=True)

    posterior = get_posterior(
        simulator, prior, summary_extractor, build_nn_posterior,
        meta_parameters, round_=round_idx, batch_theta=prior.sample((2,)), 
        batch_x=summary_extractor(torch.randn(2, 1024, 1+nextra)))

    return posterior, prior, simulator, summary_extractor

def get_samples_distance(naive, theta, round_idx=0, aggregate=True):

    samples = {}
    distance = {}

    for nextra in LIST_NEXTRA:

        # sample from the posterior distribution with naive architecture
        theta_base = torch.tensor([135.0, 220.0, 2000.0, 0.0])
        posterior, _, simulator, summary_extractor = boilerplate(
            theta_base, nextra, naive=naive, round_idx=round_idx, 
            aggregate=aggregate)

        if posterior is not None:

            xobs = simulator(theta)
            posterior.set_default_x(summary_extractor(xobs))
            samples[nextra] = posterior.sample(
                (10000,), sample_with_mcmc=False, show_progress_bars=False)

            distance[nextra] = np.zeros(len(theta))
            for j in range(len(theta)):
                samples_coordj = samples[nextra][:, j]
                sd = np.sqrt(samples_coordj.var())
                distance[nextra][j] = (
                    samples_coordj.var() + (samples_coordj.mean() - theta[j])**2
                    ) / sd

    return samples, distance

prior = prior_JRNMM(parameters=[('C', 10.0, 250.0),
                                ('mu', 50.0, 500.0),
                                ('sigma', 100.0, 5000.0),
                                ('gain', -20.0, +20.0)])

nexample = 100

list_theta = prior.sample((nexample,))
dist_matrix = np.zeros((3, 5, nexample, 4)) # 3 labels, 5 nextra, nexample, 4 coord
configs = {'naive-1':{'naive': True, 'aggregate':False},
           'naive-2':{'naive': True, 'aggregate':True},
           'factr':{'naive': False, 'aggregate':True}}

for i, labeli in enumerate(['naive-1', 'naive-2', 'factr']):
    print(labeli)
    for k, thetak in tqdm(enumerate(list_theta), total=nexample):
        naive = configs[labeli]['naive']
        aggregate = configs[labeli]['aggregate']
        _, distance = get_samples_distance(
            naive=naive, theta=thetak, round_idx=0, aggregate=aggregate)
        for j, nj in enumerate(LIST_NEXTRA):
            dist_matrix[i, j, k, :] = distance[nj]

std = [np.std(dist_matrix[:,:,:,z]) for z in range(4)]
y = np.stack([dist_matrix[:,:,:,j]/std[j] for j in range(4)])
y = np.mean(y, axis=0)

ymed = np.median(y, axis=-1) 
y025 = np.quantile(y, q=0.25, axis=-1)
y075 = np.quantile(y, q=0.75, axis=-1)      

import joblib
results = [ymed, y025, y075]
joblib.dump(results, 'results-ducktape.pkl')

# fig, ax = plt.subplots(figsize=(9.0, 6.6))
# names = ['naive', 'aggreg', 'HNPE']
# colors = ['C0', 'C1', 'C2']
# for i, label in enumerate(['naive-1', 'naive-2', 'factr']):
#     ax.plot(LIST_NEXTRA, ymed[i,:], lw=3.0, label=names[i], color=colors[i])
# ax.set_ylabel(r'Normalized $\mathcal{W}(q_{\phi}, \delta_{\theta})$', fontsize=18)
# ax.set_xlabel(r'Number of extra observations $N$', fontsize=18)
# ax.set_xticks([0, 10, 20, 30, 40])
# ax.set_xticklabels(['0', '10', '20', '30', '40'], fontsize=18)
# ax.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5])
# ax.set_yticklabels(['0.50', '0.75', '1.00', '1.25', '1.50'], fontsize=18)
# ax.legend(fontsize=16)

# fig.show()