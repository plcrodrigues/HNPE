import time
import submitit


def get_executor_jzay(job_name, timeout_hour=60, n_gpus=1, n_cpus=20):
    if timeout_hour > 20:
        qos = 't4'
    elif timeout_hour > 2:
        qos = 't3'
    else:
        qos = 'dev'

    executor = submitit.AutoExecutor(job_name)
    executor.update_parameters(
        timeout_min=180,
        slurm_job_name=job_name,
        slurm_time=f'{timeout_hour}:00:00',
        slurm_gres=f'gpu:{n_gpus}',
        slurm_additional_parameters={
            'ntasks': 1,
            'cpus-per-task': n_cpus,
            'qos': f'qos_gpu-{qos}',
            'distribution': 'block:block',
        },
        slurm_setup=[
            '#SBATCH -C v100-32g',
            'module purge',
            'module load cuda/10.1.2 '
            'cudnn/7.6.5.32-cuda-10.1 nccl/2.5.6-2-cuda'
        ]
    )
    return executor


def get_executor_marg(job_name, timeout_hour=60, n_cpus=10):

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


CLUSTER_PROFILES = {
    'jean-zay': get_executor_jzay,
    'margaret': get_executor_marg,
}


def dashboard(tasks, t_start=None, sleep_time=10, n_rounds=5):
    """Print periodical information on the state of computations.
    """
    if t_start is None:
        t_start = time.now()

    # dashboard to follow the computation evolution.
    job_id = tasks[0].job_id
    done, errored = {n_rounds: 0}, 0
    while done[n_rounds] + errored != len(tasks):
        pending, errored = 0, 0
        done = {i: 0 for i in range(n_rounds+1)}
        for i, t in enumerate(tasks):
            output = t.stdout()
            if output is None or len(output) == 0:
                pending += 1
                continue
            err = output.count('submitit ERROR') > 0
            if err:
                errored += 1
                continue
            complete_rounds = output.count(
                'Neural network successfully converged'
            )
            done[complete_rounds] += 1
            last_line = output.splitlines()[-1]
            pattern = "Training neural network."
            if pattern in last_line:
                last_line = last_line.replace(pattern, '')
                print(f'Job {i} - Round {complete_rounds + 1} - {last_line}')

        # Display general information
        fmt_done = '||'.join([
            f'Round {k}: {v / len(tasks):.1%}'
            for k, v in done.items() if v > 0
        ])
        print(
            "==========" * 6,
            f"Job #{job_id}".replace('_0', ''),
            f"Waiting: {time.time() - t_start:.0f}s",
            "----------" * 6,
            f"Pending: {pending} || {fmt_done} || Errored: {errored}",
            "==========" * 6,
            sep='\n'
        )
        time.sleep(sleep_time)
