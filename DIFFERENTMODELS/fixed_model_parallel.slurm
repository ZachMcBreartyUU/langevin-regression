#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=40
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --out="/scratch/seismology/zach/softglass/fixed_model/fixed_parallel-%j.out"
#SBATCH --error="/scratch/seismology/zach/softglass/fixed_model/fixed_parallel-%j.err"
#SBATCH --job-name FixPara
#SBATCH --exclusive

module purge
# module load development
# module load Intel/MPI/2018.3.222 # May not be needed

/eejit/home/zach/miniconda3/envs/langevin/bin/python /eejit/home/zach/LearningSoftglassEqns/Langevin_regression_method/langevin-regression/DIFFERENTMODELS/fixed_model_parameter_sweep.py