"""
Utilities for setting up experiment tracking
"""

import subprocess
from lightning.pytorch.loggers import MLFlowLogger

import params


def current_git_hash() -> str:
    cmd = "git describe --always"
    return subprocess.check_output(cmd.split()).strip().decode()


def repo_is_dirty():
    cmd = "git status . --porcelain"
    return bool(subprocess.check_output(cmd.split()).strip())


def current_git_branch() -> str:
    cmd = "git rev-parse --abbrev-ref HEAD"
    return subprocess.check_output(cmd.split()).strip().decode()


def get_logger(run_name: str):
    if repo_is_dirty():
        print(
            "Your working directory contains untracked and/or modified files. "
            "Please commit or remove your changes before starting a training run, "
            "or pass --no-track to start an untracked run."
        )
        exit()

    branch = current_git_branch()
    hash = current_git_hash()
    run_name = run_name or f"{branch}-{hash}"
    logger = MLFlowLogger(experiment_name="swedish-lion", run_name=run_name)
    hyperparams = {k: v for k, v in vars(params).items() if not k.startswith("_")}
    logger.log_hyperparams(hyperparams)
    return logger
