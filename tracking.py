"""
Utilities for setting up experiment tracking
"""

import subprocess
from lightning.pytorch.loggers import MLFlowLogger


def current_git_hash() -> str:
    cmd = "git describe --always"
    return subprocess.check_output(cmd.split()).strip().decode()


def repo_is_dirty():
    cmd = "git status . --porcelain"
    return bool(subprocess.check_output(cmd.split()).strip())


def get_logger(experiment_name: str, strict: bool):

    if strict:
        if repo_is_dirty():
            print(
                "Your working directory contains untracked and/or modified files. "
                "Please commit or remove your changes before starting a training run, "
                "or pass --no-track to start an untracked run."
            )
            exit()
        return MLFlowLogger(
            experiment_name=experiment_name, run_name=current_git_hash(), log_model=True
        )
