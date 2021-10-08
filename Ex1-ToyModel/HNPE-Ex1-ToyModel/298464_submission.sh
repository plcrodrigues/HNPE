#!/bin/bash

# Parameters
#SBATCH --array=0-8%9
#SBATCH --cpus-per-task=10
#SBATCH --distribution=block:block
#SBATCH --error=/data/parietal/store/work/pcoelhor/development/HNPE/Ex1-ToyModel/HNPE-Ex1-ToyModel/%A_%a_0_log.err
#SBATCH --job-name=HNPE_Ex1_ToyModel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/data/parietal/store/work/pcoelhor/development/HNPE/Ex1-ToyModel/HNPE-Ex1-ToyModel/%A_%a_0_log.out
#SBATCH --signal=USR1@90
#SBATCH --time=72:00:00
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --output /data/parietal/store/work/pcoelhor/development/HNPE/Ex1-ToyModel/HNPE-Ex1-ToyModel/%A_%a_%t_log.out --error /data/parietal/store/work/pcoelhor/development/HNPE/Ex1-ToyModel/HNPE-Ex1-ToyModel/%A_%a_%t_log.err --unbuffered /data/parietal/store/work/pcoelhor/miniconda3/bin/python -u -m submitit.core._submit /data/parietal/store/work/pcoelhor/development/HNPE/Ex1-ToyModel/HNPE-Ex1-ToyModel