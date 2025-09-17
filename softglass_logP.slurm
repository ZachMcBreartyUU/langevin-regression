#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --out="/scratch/seismology/zach/runs_EqnLearning/LangevinRegression-%j.out"
#SBATCH --error="/scratch/seismology/zach/runs_EqnLearning/LangevinRegression-%j.err"
#SBATCH --job-name LangevinP

module purge

/eejit/home/zach/miniconda3/envs/langevin/bin/python /eejit/home/zach/LearningSoftglassEqns/Langevin_regression_method/langevin-regression/softglass_logP.py "$@"