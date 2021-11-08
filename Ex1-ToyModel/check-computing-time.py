from functools import partial

import torch

from hnpe.misc import make_label
from hnpe.inference import run_inference
import matplotlib.pyplot as plt

from viz import get_posterior
from viz import display_posterior
from posterior import build_flow, IdentityToyModel
from simulator import simulator_ToyModel, prior_ToyModel, get_ground_truth

nrd = 1
nsr = 10_000
maxepochs = 150
stop_after_epochs = 151
saverounds = False

nextra = 100
alpha = 0.50
beta = 0.50
gamma = 1.00

for naive in [False, True]:

    # setup the parameters for the example
    meta_parameters = {}
    # how many extra observations to consider
    meta_parameters["n_extra"] = nextra
    # how many trials for each observation
    meta_parameters["n_trials"] = 1
    # what kind of summary features to use
    meta_parameters["summary"] = 'Identity'
    # the parameters of the ground truth (observed data)
    meta_parameters["theta"] = torch.tensor([alpha, beta])
    # gamma parameter on x = alpha * beta^gamma + w
    meta_parameters["gamma"] = gamma
    # standard deviation of the noise added to the observations
    meta_parameters["noise"] = 0.00
    # which example case we are considering here
    meta_parameters["case"] = ''.join([
        f"CheckComputingTime/ToyModel_",
        f"naive_{naive}_",
        f"ntrials_{meta_parameters['n_trials']:02}_",
        f"nextra_{meta_parameters['n_extra']:02}_",
        f"alpha_{meta_parameters['theta'][0]:.2f}_",
        f"beta_{meta_parameters['theta'][1]:.2f}_",
        f"gamma_{meta_parameters['gamma']:.2f}_",
        f"noise_{meta_parameters['noise']:.2f}"])
    # number of rounds to use in the SNPE procedure
    meta_parameters["n_rd"] = nrd
    # number of simulations per round
    meta_parameters["n_sr"] = nsr
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

    # run inference procedure over the example
    posteriors = run_inference(simulator=simulator,
                                prior=prior,
                                build_nn_posterior=build_nn_posterior,
                                ground_truth=ground_truth,
                                meta_parameters=meta_parameters,
                                summary_extractor=summary_net,
                                save_rounds=saverounds,
                                device='cpu',
                                max_num_epochs=maxepochs,
                                stop_after_epochs=stop_after_epochs)

    posterior = posteriors[0]