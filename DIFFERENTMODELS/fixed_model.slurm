#!/bin/bash
#SBATCH --ntasks=9
#SBATCH --ntasks-per-node=9
#SBATCH --threads-per-core=1
#SBATCH --nodes=1
#SBATCH --out="/scratch/seismology/zach/softglass/fixed_model/Batch-%j.out"
#SBATCH --error="/scratch/seismology/zach/softglass/fixed_model/Batch-%j.err"
#SBATCH --job-name Batch
#SBATCH --exclusive

module purge
module load development
module load Intel/MPI/2018.3.222 # May not be needed

mpiexec -n 9 /eejit/home/zach/miniconda3/envs/langevin/bin/python /eejit/home/zach/LearningSoftglassEqns/Langevin_regression_method/langevin-regression/DIFFERENTMODELS/fixed_model.py
# /eejit/home/zach/miniconda3/envs/langevin/bin/python /eejit/home/zach/LearningSoftglassEqns/Langevin_regression_method/langevin-regression/DIFFERENTMODELS/fixed_model.py