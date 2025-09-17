import subprocess

# relative to SCRATCH_PATH in data_loader.py
EQN_LEARN_PATH = "runs_EqnLearning/"

for foldername in [
    "run_0D_0D_5_1_const_3",
    "run_0D_0D_5_1_const_4",
    "run_0D_0D_5_1_const_35",
    "run_0D_0D_5_1_const_37",
]:
    folder = EQN_LEARN_PATH + foldername
    subprocess.run(
        [
            "sbatch",
            "-o",
            f"/scratch/seismology/zach/{folder}.out",
            "-e",
            f"/scratch/seismology/zach/{folder}.err",
            "/eejit/home/zach/LearningSoftglassEqns/Langevin_regression_method/langevin-regression/process_data.slurm",
            folder,
            "-N",
            "101",
            "-c",
            "-25",
            "--log-spacing",
        ]
    )
