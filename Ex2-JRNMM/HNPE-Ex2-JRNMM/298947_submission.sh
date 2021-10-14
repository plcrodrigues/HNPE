#!/bin/bash

# Parameters
#SBATCH --array=0-4%5
#SBATCH --cpus-per-task=40
#SBATCH --distribution=block:block
#SBATCH --error=/data/parietal/store/work/pcoelhor/development/HNPE/Ex2-JRNMM/HNPE-Ex2-JRNMM/%A_%a_0_log.err
#SBATCH --job-name=HNPE_Ex2_JRNMM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/data/parietal/store/work/pcoelhor/development/HNPE/Ex2-JRNMM/HNPE-Ex2-JRNMM/%A_%a_0_log.out
#SBATCH --signal=USR1@90
#SBATCH --time=60:00:00
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --output /data/parietal/store/work/pcoelhor/development/HNPE/Ex2-JRNMM/HNPE-Ex2-JRNMM/%A_%a_%t_log.out --error /data/parietal/store/work/pcoelhor/development/HNPE/Ex2-JRNMM/HNPE-Ex2-JRNMM/%A_%a_%t_log.err --unbuffered /data/parietal/store/work/pcoelhor/miniconda3/bin/python -u -m submitit.core._submit /data/parietal/store/work/pcoelhor/development/HNPE/Ex2-JRNMM/HNPE-Ex2-JRNMM