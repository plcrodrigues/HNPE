from functools import partial
import torch
from hnpe.misc import make_label
from hnpe.inference import run_inference

from posterior import build_flow, IdentityToyModel
from simulator import simulator_ToyModel_amortizeNextra, prior_ToyModel, get_ground_truth

from torch.distributions import Categorical

import submitit     

LIST_NEXTRA_RANGE = [40]
LIST_NORM_BEFORE = [False]

NSR = 10_000

def get_executor_marg(job_name, timeout_hour=60, n_cpus=40):

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

def setup_inference(nextra_range, norm_before, num_workers=20):
       
    # setup the parameters for the example
    meta_parameters = {}
    # the range of values of how many extra observations to consider
    meta_parameters["nextra_range"] = nextra_range
    #aggregation method for extra observations 
    meta_parameters["aggregate_method"] = 'mean'

    # what kind of summary features to use
    meta_parameters["summary"] = 'Identity'
   
    # gamma parameter on x = alpha * beta^gamma + w
    meta_parameters["gamma"] = 1.0
    # standard deviation of the noise added to the observations
    meta_parameters["noise"] = 0.0
    # whether to do naive implementation
    meta_parameters["naive"] = True
    # how many trials for each observation
    meta_parameters["n_trials"] = 1
    # whether to normalize x0 and xn before training intead of 
    # zscore on x0,nextra,xn during training 
    meta_parameters["norm_before"] = norm_before
    # which example case we are considering here
    meta_parameters["case"] = "Flow/2nd_run_ToyModel_nextra_range_{:02}_" \
                    "naive_{}_aggregate_{}".format(meta_parameters["nextra_range"],
                        meta_parameters["naive"], meta_parameters["aggregate_method"])

    # number of rounds to use in the SNPE procedure
    meta_parameters["n_rd"] = 1
    # number of simulations per round
    meta_parameters["n_sr"] = NSR
    # number of summary features to consider
    meta_parameters["n_sf"] = 1

    # label to attach to the SNPE procedure and use for saving files
    meta_parameters["label"] = make_label(meta_parameters)
    # run example with the chosen parameters
    device = "cpu"

    # set probabilities for categorical distribution on nextra values 
    nextra_probs = torch.ones(meta_parameters['nextra_range'])*1/meta_parameters['nextra_range']
    prior_nextra = Categorical(nextra_probs)

    # set prior distribution for the parameters
    prior = prior_ToyModel(low=torch.tensor([0.0, 0.0]),
                           high=torch.tensor([1.0, 1.0])
                           )

    # choose how to setup the simulator
    simulator = partial(simulator_ToyModel_amortizeNextra,
                        n_trials=meta_parameters["n_trials"],
                        p_alpha=prior,
                        p_nextra=prior_nextra,
                        gamma=meta_parameters["gamma"],
                        sigma=meta_parameters["noise"],
                        aggregate_method=meta_parameters["aggregate_method"])                    

    # choose how to get the summary features
    summary_net = IdentityToyModel()       

    # choose a function which creates a neural network density estimator
    build_nn_posterior = partial(build_flow,
                                 embedding_net=summary_net,
                                 naive=meta_parameters["naive"], ## for now always True
                                 z_score_x=not meta_parameters["norm_before"])  

    _ = run_inference(simulator=simulator, 
                    prior=prior, 
                    build_nn_posterior=build_nn_posterior, 
                    ground_truth=None,
                    meta_parameters=meta_parameters, 
                    summary_extractor=summary_net, 
                    save_rounds=True,
                    device=device, num_workers=num_workers,
                    max_num_epochs=None)    

executor = get_executor_marg(f'work_inference')
# launch batches
with executor.batch():
    print('Submitting jobs...', end='', flush=True)
    tasks = []   
    for nextra_range in LIST_NEXTRA_RANGE:
        for norm_before in LIST_NORM_BEFORE:
            kwargs = {'nextra_range':nextra_range, 'norm_before':norm_before} 
            tasks.append(executor.submit(setup_inference, **kwargs))  
             
