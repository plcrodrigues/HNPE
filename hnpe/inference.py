import torch
import numpy as np
from pathlib import Path

from sbi import inference as sbi_inference
from sbi.utils import get_log_root
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def summary_plcr(prefix):
    logdir = Path(
        get_log_root(),
        prefix,
        datetime.now().isoformat().replace(":", "_"),
    )
    return SummaryWriter(logdir)


def run_inference(simulator, prior, build_nn_posterior, ground_truth,
                  meta_parameters, summary_extractor=None, save_rounds=False,
                  seed=42, device="cpu", num_workers=1, max_num_epochs=None,
                  stop_after_epochs=20, training_batch_size=100):

    # set seed for numpy and torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    # make a SBI-wrapper on the simulator object for compatibility
    simulator, prior = sbi_inference.prepare_for_sbi(simulator, prior)

    if save_rounds:
        # save the ground truth and the parameters
        folderpath = Path.cwd() / "results" / meta_parameters["label"]
        print(folderpath)
        folderpath.mkdir(exist_ok=True, parents=True)
        path = folderpath / "ground_truth.pkl"
        torch.save(ground_truth, path)
        path = folderpath / "parameters.pkl"
        torch.save(meta_parameters, path)

    # setup the inference procedure
    inference = sbi_inference.SNPE(
        prior=prior,
        density_estimator=build_nn_posterior,
        show_progress_bars=True,
        device=device,
        summary_writer=summary_plcr(meta_parameters["label"])
    )

    # loop over rounds
    posteriors = []
    proposal = prior
    for round_ in range(meta_parameters["n_rd"]):

        # simulate the necessary data
        theta, x = sbi_inference.simulate_for_sbi(
            simulator, proposal, num_simulations=meta_parameters["n_sr"],
            num_workers=num_workers,
        )
        if 'cuda' in device:
            torch.cuda.empty_cache()

        # extract summary features
        if summary_extractor is not None:
            x = summary_extractor(x)

        # train the neural posterior with the loaded data
        nn_posterior = inference.append_simulations(theta, x).train(
            num_atoms=10,
            training_batch_size=training_batch_size,
            use_combined_loss=True,
            discard_prior_samples=True,
            max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs,
            show_train_summary=True
        )
        nn_posterior.zero_grad()
        posterior = inference.build_posterior(nn_posterior)
        posteriors.append(posterior)
        # save the parameters of the neural posterior for this round
        if save_rounds:
            path = folderpath / f"nn_posterior_round_{round_:02}.pkl"
            posterior.net.save_state(path)

        # set the proposal prior for the next round
        proposal = posterior.set_default_x(ground_truth['observation'])

    return posteriors
