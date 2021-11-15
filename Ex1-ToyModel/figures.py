
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

THETA_LIST = [(0.50, 0.50), (0.25, 0.75), (0.75, 0.25), 
              (0.75, 0.75), (0.25, 0.25), (0.90, 0.50), 
              (0.50, 0.10), (0.10, 0.50), (0.50, 0.90)]

NEXTRA_LIST = list(np.unique(np.logspace(0, 3, 20, dtype=int)))   

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

def get_results_our_proposal(naive):

    distance = {}
    for nextra in NEXTRA_LIST:
        distance[nextra] = []

    for nextra in tqdm(NEXTRA_LIST):

        for theta in THETA_LIST:
            
            alpha, beta = theta
            posterior = get_posterior(
                alpha=alpha, beta=beta, gamma=1.00, 
                nextra=nextra, ntrials=1, noise=0.00, 
                naive=naive, round_idx=4)
            samples = posterior.sample((10_000,), 
                sample_with_mcmc=False,
                show_progress_bars=False)

            dist_var = samples.var(dim=0).numpy()
            dist_mean = (samples.mean(dim=0) - np.array([alpha, beta]))**2
            dist = dist_mean + dist_var

            distance[nextra].append(dist.numpy())

    distance_quantiles = np.stack([np.stack([np.quantile(np.stack(distance[nextra])[:,i], q=[0.25, 0.50, 0.75]) for i in range(2)]) for nextra in NEXTRA_LIST])
    distance_quantiles_alpha = distance_quantiles[:,0,:]
    distance_quantiles_beta = distance_quantiles[:,1,:]        

    return distance_quantiles_alpha, distance_quantiles_beta

def get_results_hABC():

    distance = {}
    for nextra in NEXTRA_LIST:
        distance[nextra] = []

    for nextra in tqdm(NEXTRA_LIST):
        for theta in THETA_LIST:
            alpha, beta = theta
            gamma = 1.0
            noise = 0.00
            filename = f"./results/MCABC/ToyModel_MCABC_ntrials_01_nextra_{nextra:02}_"
            filename = filename + f"alpha_{alpha:.2f}_beta_{beta:.2f}_gamma_{gamma:.2f}_noise_{noise:.2f}/"
            filename = filename + f"Identity_n_rd_1_n_sr_50000_n_sf_1/posterior_n_sr_50000.pkl"
            posterior = torch.load(filename)
            samples = posterior.sample((10000,))
            dist_var = samples.var(dim=0).numpy()
            dist_mean = (samples.mean(dim=0) - np.array([alpha, beta]))**2
            dist = dist_mean + dist_var
            distance[nextra].append(dist.numpy())

    distance_quantiles = np.stack([np.stack([np.quantile(np.stack(distance[nextra])[:,i], q=[0.25, 0.50, 0.75]) for i in range(2)]) for nextra in NEXTRA_LIST])
    distance_quantiles_alpha = distance_quantiles[:,0,:]
    distance_quantiles_beta = distance_quantiles[:,1,:]               

    return distance_quantiles_alpha, distance_quantiles_beta

def get_results_Tran2017():

    distance = {}
    for nextra in NEXTRA_LIST:
        distance[nextra] = []

    for nextra in tqdm(NEXTRA_LIST):
        for theta in THETA_LIST:
            alpha, beta = theta
            filename = f"./results/Edward/alpha_{alpha:.2f}_beta_{beta:.2f}/"
            filename = filename + f'N_{nextra:03}_nitratio_50_nitvi_01.pkl'
            data = joblib.load(filename)
            samples_alpha = data['samples']['x0'][1]
            samples_beta = data['samples']['w'][1]
            samples = torch.tensor(np.concatenate([samples_alpha, samples_beta], axis=1))        
            dist_var = samples.var(dim=0).numpy()
            dist_mean = (samples.mean(dim=0) - np.array([alpha, beta]))**2
            dist = dist_mean + dist_var
            distance[nextra].append(dist.numpy()) 

    distance_quantiles = np.stack([np.stack([np.quantile(np.stack(distance[nextra])[:,i], q=[0.25, 0.50, 0.75]) for i in range(2)]) for nextra in NEXTRA_LIST])
    distance_quantiles_alpha = distance_quantiles[:,0,:]
    distance_quantiles_beta = distance_quantiles[:,1,:]

    return distance_quantiles_alpha, distance_quantiles_beta

distance = {}
distance['alpha'] = {}
distance['beta'] = {}

distance['alpha']['naive'], distance['beta']['naive'] = get_results_our_proposal(naive=True)   
distance['alpha']['HNPE'], distance['beta']['HNPE'] = get_results_our_proposal(naive=False) 
distance['alpha']['h-ABC'], distance['beta']['h-ABC'] = get_results_hABC()
distance['alpha']['LFVI'], distance['beta']['LFVI'] = get_results_Tran2017()

fig, ax = plt.subplots(figsize=(11.3, 5.1), ncols=2)
plt.subplots_adjust(wspace=0.20, bottom=0.15, top=0.90)

colors = {'h-ABC':'C0', 'LFVI':'C1', 'naive':'C2', 'HNPE':'C2'}
for i, coord in enumerate(['alpha', 'beta']):
    for label in ['HNPE', 'naive', 'h-ABC', 'LFVI']:
        if label == 'naive':
            ls = '--'
        else:
            ls = '-'
        ax[i].plot(NEXTRA_LIST, distance[coord][label][:,1], lw=2.0, label=label, c=colors[label], ls=ls)
        # ax[i].fill_between(NEXTRA_LIST, distance[coord][label][:,0], distance[coord][label][:,2], alpha=0.3, color=colors[label])
ax[0].set_title(r'$\mathcal{W}(\alpha_0)$', fontsize=16)
ax[1].set_title(r'$\mathcal{W}(\beta)$', fontsize=16)

for axi in ax:
    axi.set_xscale('log')
    axi.set_yscale('log')
    axi.set_xlim(1, 1000)
    axi.set_xlabel(r'Number of observations $N$')
ax[1].legend(loc="lower left")
fig.show()