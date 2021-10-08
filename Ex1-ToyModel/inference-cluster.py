from functools import partial
import torch
from sbi_workforce.misc import make_label
from sbi_workforce.inference import run_inference
import numpy as np
from posterior import build_flow, IdentityToyModel
from simulator import simulator_ToyModel, prior_ToyModel, get_ground_truth
import submitit

LIST_ALPHA_BETA = [(0.5, 0.5), (0.25, 0.75), (0.75, 0.25),
                   (0.75, 0.75), (0.25, 0.25), (0.9, 0.5),
                   (0.5, 0.1), (0.1, 0.5), (0.5, 0.9)]

LIST_NEXTRA = [0] + list(np.unique(np.logspace(0, 3, 20, dtype=int)))


def get_executor_cluster_margaret(job_name, timeout_hour=72, n_cpus=10):

    executor = submitit.AutoExecutor(job_name)
    executor.update_parameters(
        timeout_min=180,
        slurm_job_name=job_name,
        slurm_time=f'{timeout_hour}:00:00',
        slurm_additional_parameters={
            'ntasks': 1,
            'cpus-per-task': n_cpus,
            'distribution': 'block:block',
        },
    )
    return executor


def setup_inference(noise, alpha, beta):

    for nextra in LIST_NEXTRA:

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
            meta_parameters["gamma"] = 1.00
            # standard deviation of the noise added to the observations
            meta_parameters["noise"] = noise
            # which example case we are considering here
            meta_parameters["case"] = ''.join([
                f"ToyModel_",
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
            prior = prior_ToyModel(
                low=torch.tensor([0.0, 0.0]),
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

            # choose a function which creates a neural network estimator
            build_nn_posterior = partial(build_flow,
                                         embedding_net=summary_net,
                                         naive=naive)

            _ = run_inference(simulator=simulator,
                              prior=prior,
                              build_nn_posterior=build_nn_posterior,
                              ground_truth=ground_truth,
                              meta_parameters=meta_parameters,
                              summary_extractor=summary_net,
                              save_rounds=True,
                              device='cpu',
                              max_num_epochs=None)


executor = get_executor_cluster_margaret(f'HNPE-Ex1-ToyModel')
# launch batches
with executor.batch():
    print('Submitting jobs...', end='', flush=True)
    tasks = []
    # loop through difference ground truth parameters
    for params in LIST_ALPHA_BETA:
        alpha, beta = params
        # results on different noise levels
        kwargs = {'noise': 0.00,
                  'alpha': alpha,
                  'beta': beta}
        tasks.append(executor.submit(setup_inference, **kwargs))
