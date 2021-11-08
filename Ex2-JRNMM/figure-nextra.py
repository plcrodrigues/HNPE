
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

LIST_THETA = [[135.0, 220.0, 2000.0, 0.0],
              [135.0, 220.0, 2000.0, -10.0],
              [135.0, 220.0, 2000.0, 10.0],
              [270.0, 220.0, 2000.0, 0.0],
              [68.0, 220.0, 2000.0, 0.0],
              [135.0, 110.0, 2000.0, 0.0],
              [68.0, 220.0, 2000.0, 10.0],
              [270.0, 220.0, 2000.0, 10.0],
              [135.0, 110.0, 1000.0, 0.0],
              [135.0, 220.0, 1000.0, 0.0]
              ]

def boilerplate(theta, nextra, naive, aggregate=True, round_idx=0):

    # setup the parameters for the example
    meta_parameters = {}
    # how many extra observations to consider
    meta_parameters["n_extra"] = nextra
    # what kind of summary features to use
    meta_parameters["summary"] = 'Fourier'
    # the parameters of the ground truth (observed data)
    meta_parameters["theta"] = torch.tensor(theta)

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

    try:
        posterior = get_posterior(
            simulator, prior, summary_extractor, build_nn_posterior,
            meta_parameters, round_=round_idx, batch_theta=prior.sample((2,)), 
            batch_x=summary_extractor(torch.randn(2, 1024, 1+nextra))
        )
    except:
        print(f"Problem at {meta_parameters['case']}")
        posterior = None

    return posterior, prior

def get_samples_distance(naive, theta, round_idx=1, aggregate=True):

    samples = {}
    distance = {}
    for nextra in LIST_NEXTRA:
        # sample from the posterior distribution with naive architecture
        posterior, _ = boilerplate(
            theta, nextra, naive=naive, round_idx=round_idx, aggregate=aggregate)
        if posterior is not None:
            samples[nextra] = posterior.sample(
                (10000,), sample_with_mcmc=False)

            distance[nextra] = np.zeros(len(theta))
            for j in range(len(theta)):
                samples_coordj = samples[nextra][:, j]
                sd = np.sqrt(samples_coordj.var())
                distance[nextra][j] = (
                    samples_coordj.var() + (samples_coordj.mean() - theta[j])**2
                    ) / sd

    return samples, distance

def get_median_distance(dist):

    dist_med = {}
    for nextra in LIST_NEXTRA:
        dist_med[nextra] = []
    for i in range(len(dist)):
        for nextra in LIST_NEXTRA:
            if nextra in dist[i]:
                dist_med[nextra].append(dist[i][nextra])
    for nextra in LIST_NEXTRA:
        dist_med[nextra] = np.median(dist_med[nextra], axis=0)

    return dist_med

round_idx = 0

dist_dic = {}

configs = {'naive-1':{'naive': True, 'aggregate':False},
           'naive-2':{'naive': True, 'aggregate':True},
           'factr':{'naive': False, 'aggregate':True}}

for label in ['naive-1', 'naive-2', 'factr']:
    dist_dic[label] = []
    for thetai in LIST_THETA:
        _, distance = get_samples_distance(
            naive=configs[label]['naive'], 
            theta=thetai, 
            round_idx=round_idx, 
            aggregate=configs[label]['aggregate'])
        if len(distance) == len(LIST_NEXTRA):
            dist_dic[label].append(distance)

# fig, ax = plt.subplots(figsize=(10.25, 5.15))
# plt.subplots_adjust(left=0.175, right=0.95, bottom=0.15, top=0.95)
# for label in ['naive-1', 'naive-2', 'factr']:
#     ax.plot(LIST_NEXTRA, dist_dic[label], lw=2.0, label=label)
# # ax.set_ylim(-0.05, 1.05)
# ax.set_ylabel(r'$\mathcal{W}(q_{\phi}, \delta_{\theta})$')
# ax.set_xlabel(r'Number of extra observations $N$')
# ax.set_xticks([0, 10, 20, 30, 40])
# ax.set_xticklabels(['0', '10', '20', '30', '40'], fontsize=18)
# # ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
# # ax.set_yticklabels(['0.0', '0.25', '0.50', '0.75', '1.00'], fontsize=18)
# ax.legend(fontsize=16)
# # plt.savefig(f'figure_round_{round_idx}.pdf', format='pdf')
# fig.show()

fig, ax = plt.subplots(figsize=(12.4, 11.4), ncols=2, nrows=2)
parameters = ['$C$', '$\mu$', '$\sigma$', '$g$']
for label in ['naive-1', 'naive-2', 'factr']:
    for z, axz in enumerate(ax.flatten()):
        dist_matrix = np.zeros((len(LIST_NEXTRA), len(dist_dic[label])))
        for i in range(dist_matrix.shape[0]):
            for j in range(dist_matrix.shape[1]):
                dist_matrix[i,j] = dist_dic[label][j][LIST_NEXTRA[i]][z]
        y = np.median(dist_matrix, axis=1)
        axz.plot(LIST_NEXTRA, y, lw=3.0, label=label)
        if z > 1:
            axz.set_xlabel('$N$', fontsize=14)
        axz.set_title(parameters[z], fontsize=18)
axz.legend()
fig.show()

ntheta = 8
dist_matrix = np.zeros((3, 5, ntheta, 4)) # 3 labels, 5 nextra, ntheta, 4 coord
for i, label in enumerate(['naive-1', 'naive-2', 'factr']): # loop labels
    for j in range(5): # loop nextra
        for k in range(ntheta): # loop theta
            for l in range(4): # loop coordinates
                dist_matrix[i, j, k, l] = dist_dic[label][k][LIST_NEXTRA[j]][l]

std = [np.std(dist_matrix[:,:,:,z]) for z in range(4)]
y = np.stack([dist_matrix[:,:,:,j]/std[j] for j in range(4)])
y = np.mean(y, axis=0)
ymed = np.median(y, axis=-1)
y025 = np.quantile(y, q=0.25, axis=-1)
y075 = np.quantile(y, q=0.75, axis=-1)
fig, ax = plt.subplots(figsize=(8.3, 7.7))
for i, label in enumerate(['naive-1', 'naive-2', 'factr']):
    ax.plot(LIST_NEXTRA, ymed[i,:], lw=3.0, label=label)
    # ax.fill_between(LIST_NEXTRA, y025[i,:], y075[i,:], alpha=0.10)
ax.legend()
fig.show()