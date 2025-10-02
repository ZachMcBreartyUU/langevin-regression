import subprocess

# relative to SCRATCH_PATH in data_loader.py
EQN_LEARN_PATH = "runs_EqnLearning/"

for foldername in [
    # "run_0D_0D_5_1_const_3",
    # "run_0D_0D_5_1_const_4",
    # "run_0D_0D_5_1_const_35",
    # "run_0D_0D_5_1_const_37",
    "run_0D_0D_5_1_000",
    # "run_0D_0D_5_1_004",
    # "run_0D_0D_6_1_000",
    # "run_0D_0D_6_1_004",
]:
    folder = EQN_LEARN_PATH + foldername
    subprocess.run(
        [
            "sbatch",
            "-o",
            f"/scratch/seismology/zach/{folder}/process_data_2D.out",
            "-e",
            f"/scratch/seismology/zach/{folder}/process_data_2D.err",
            "/eejit/home/zach/LearningSoftglassEqns/Langevin_regression_method/langevin-regression/process_data_2D.slurm",
            folder,
            "-Nf",
            "100",
            "-Ns",
            "20",
            "-c",
            "-10",
            "--log-spacing-fluidity",
            "--plot-intermediate",
        ]
    )
