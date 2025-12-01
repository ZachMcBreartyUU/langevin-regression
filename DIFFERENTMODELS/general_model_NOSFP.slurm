#!/bin/bash
#SBATCH --ntasks=9
#SBATCH --ntasks-per-node=9
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --out="/scratch/seismology/zach/runs_EqnLearning/LangevinRegression-%j.out"
#SBATCH --error="/scratch/seismology/zach/runs_EqnLearning/LangevinRegression-%j.err"
#SBATCH --job-name Pitchfork
#  #SBATCH --exclusive

module purge
# module load development
# module load Intel/MPI/2018.3.222 # May not be needed

/eejit/home/zach/miniconda3/envs/langevin/bin/python /eejit/home/zach/LearningSoftglassEqns/Langevin_regression_method/langevin-regression/DIFFERENTMODELS/general_model_stacking_K1_Alpha_no_SFP.py "$@" --PARALLEL