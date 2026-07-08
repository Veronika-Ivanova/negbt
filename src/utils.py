""""
Utils.
"""

import os
import random
import numpy as np
from glob import glob
import pytorch_lightning as pl
import torch


import pandas as pd


def extract_validation_history(path):

    events_path = glob(os.path.join(path, 'events.*'))[0]

    event_acc = EventAccumulator(events_path)
    event_acc.Reload()

    scalars = event_acc.Tags()['scalars']
    history = pd.DataFrame(columns=['step'])
    for scalar in scalars:
        events = event_acc.Scalars(tag=scalar)
        df_scalar = pd.DataFrame(
            [(event.step, event.value) for event in events], columns=['step', scalar])
        history = pd.merge(history, df_scalar, on='step', how='outer')

    return history

def fix_seed(seed):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_rng_state


def get_csv_metrics_path(trainer, filename='metrics.csv'):
    """Return path to CSV metrics file (PL 1.x and 2.x CSVLogger)."""
    logger = trainer.logger
    if isinstance(logger, (list, tuple)):
        logger = logger[0]

    if hasattr(logger, 'log_dir'):
        log_dir = logger.log_dir
        if not callable(log_dir):
            return os.path.join(log_dir, filename)

    experiment = logger.experiment
    log_dir = experiment.log_dir
    if callable(log_dir):
        log_dir = log_dir()
    return os.path.join(log_dir, filename)
