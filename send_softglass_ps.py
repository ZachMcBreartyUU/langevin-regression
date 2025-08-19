import subprocess

# relative to SCRATCH_PATH in data_loader.py
EQN_LEARN_PATH = "runs_EqnLearning/"

for foldername in [
    "run_0D_0D_5_1_000",
    "run_0D_0D_5_1_004",
    "run_0D_0D_6_1_000",
    "run_0D_0D_6_1_004",
]:
    folder = EQN_LEARN_PATH + foldername
    subprocess.run(
        [
            "sbatch",
            "-o",
            f"/scratch/seismology/zach/{folder}_ps_redlib_anm.out",
            "-e",
            f"/scratch/seismology/zach/{folder}_ps_redlib_anm.err",
            "/eejit/home/zach/LearningSoftglassEqns/Langevin_regression_method/langevin-regression/softglass_phi_and_sigma.slurm",
            folder,
        ]
    )
