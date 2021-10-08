from functools import partial
import torch
from sbi_workforce.misc import make_label
from sbi_workforce.inference import run_inference
from posterior import build_flow, IdentityJRNMM
from summary import summary_JRNMM
from simulator import prior_JRNMM, simulator_JRNMM, get_ground_truth
import submitit

LIST_THETA = [[135.0, 220.0, 2000.0, 0.0],
              [135.0, 220.0, 2000.0, -10.0],
              [135.0, 220.0, 2000.0, 10.0],
              [270.0, 220.0, 2000.0, 0.0],
              [68.0, 220.0, 2000.0, 0.0]]

LIST_NEXTRA = [0, 10, 20, 30, 40]

def get_executor_cluster_margaret(job_name, timeout_hour=72, n_cpus=40):

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

def setup_inference(theta, naive, num_workers=20):
            
    for nextra in LIST_NEXTRA:
        
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
            # run example with the chosen parameters
            device = "cpu"

            # set prior distribution for the parameters
            input_parameters = ['C', 'mu', 'sigma', 'gain']
            prior = prior_JRNMM(parameters=[('C', 10.0, 250.0),
                                            ('mu', 50.0, 500.0),
                                            ('sigma', 100.0, 5000.0),
                                            ('gain', -20.0, +20.0)])

            # choose how to setup the simulator
            simulator = partial(
                simulator_JRNMM,
                input_parameters=input_parameters,
                t_recording=meta_parameters["t_recording"],
                n_extra=meta_parameters["n_extra"],
                p_gain=prior)                         

            # choose how to get the summary features
            summary_extractor = summary_JRNMM(
                n_extra=meta_parameters["n_extra"],
                d_embedding=meta_parameters["n_sf"],
                n_time_samples=meta_parameters["n_ss"],
                type_embedding=meta_parameters["summary"])

            # let's use the log power spectral density instead
            summary_extractor.embedding.net.logscale = True       

            # choose the ground truth observation to consider in the inference
            ground_truth = get_ground_truth(
                meta_parameters,
                input_parameters=input_parameters,
                p_gain=prior)  
            ground_truth['observation'] = summary_extractor(
                ground_truth['observation'])

            # choose a function which creates a neural density estimator
            build_nn_posterior = partial(
                build_flow, 
                embedding_net=IdentityJRNMM(),
                naive=meta_parameters["naive"],
                aggregate=True,
                z_score_theta=True,
                z_score_x=True)  

            _ = run_inference(
                    simulator=simulator, 
                    prior=prior, 
                    build_nn_posterior=build_nn_posterior, 
                    ground_truth=ground_truth,
                    meta_parameters=meta_parameters, 
                    summary_extractor=summary_extractor, 
                    save_rounds=True,
                    device=device, num_workers=num_workers,
                    max_num_epochs=None)    

executor = get_executor_cluster_margaret(f'HNPE-Ex2-JRNMM')
# launch batches
with executor.batch():
    print('Submitting jobs...', end='', flush=True)
    tasks = []   
    # loop on increasing numbers of nextra values
    for theta in LIST_THETA:
        for naive in [False, True]:
            kwargs = {'theta':theta, 'naive':naive} 
            tasks.append(executor.submit(setup_inference, **kwargs))              
