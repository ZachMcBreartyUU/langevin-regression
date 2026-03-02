#!/bin/bash
#SBATCH --ntasks=15
#SBATCH --ntasks-per-node=15
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --out="/scratch/seismology/zach/softglass/SSR_comp_stats.out"
#SBATCH --error="/scratch/seismology/zach/softglass/SSR_comp_stats.err"
#SBATCH --job-name SSRCOMPStats
# SBATCH --exclusive

module purge
# module load development
# module load Intel/MPI/2018.3.222 # May not be needed

/eejit/home/zach/miniconda3/envs/langevin/bin/python /eejit/home/zach/LearningSoftglassEqns/Langevin_regression_method/langevin-regression/DIFFERENTMODELS/SSR_Compare_methods_stats.py