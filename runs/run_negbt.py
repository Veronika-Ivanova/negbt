import logging
import os
import random
import sys

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader

from datasets import (
    PaddingCollateFn,
    BarlowDataset,
    CausalPredictionDataset,
)
from modules import SeqRec, SeqRecBarlow
from models import SASRec, SASRecBarlow
from postprocess import preds2recs
from preprocess import add_time_idx, prepare_splitted_data
from metrics import compute_metrics
from utils import extract_validation_history, fix_seed

cfg = OmegaConf.load("../runs/configs/NegBT_ml.yaml")

PROJECT_PATH = f"{cfg.project_path}"
DATA_PATH = f"{PROJECT_PATH}{cfg.data_path}"

USER_COL = f"{cfg.dataset.user_col}"
ITEM_COL = f"{cfg.dataset.item_col}"
RELEVANCE_COL = f"{cfg.dataset.relevance_col}"
RELEVANCE_THRESHOLD = cfg.dataset.relevance_threshold
MAX_LENGTH = cfg.dataset.max_length
only_positive = cfg.dataset.only_positive

VALIDATION_SIZE = cfg.dataloader.validation_size
BATCH_SIZE = cfg.dataloader.batch_size
TEST_BATCH_SIZE = cfg.dataloader.test_batch_size
NUM_WORKERS = cfg.dataloader.num_workers
SEED = cfg.dataloader.seed

dropout = cfg.model.dropout
hidden_units = cfg.model.hidden_units
lr = cfg.model.lr
num_blocks = cfg.model.num_blocks
barlow_coeff = cfg.model.barlow_coeff
off_diag_coeff = cfg.model.off_diag_coeff
power_coef = cfg.model.power_coef


fix_seed(SEED)
train, train_full, validation, test_pos, test_neg, last_item_pos, last_item_neg = prepare_splitted_data(
    DATA_PATH,
    user_col=USER_COL,
    relevance_col=RELEVANCE_COL,
    filter_negative=False,
    relevance_threshold=RELEVANCE_THRESHOLD,
)


def get_eval_dataset(validation, validation_size=VALIDATION_SIZE):
    validation_users = validation.user_id.unique()

    if validation_size and (validation_size < len(validation_users)):
        validation_users = np.random.choice(
            validation_users, size=validation_size, replace=False
        )

    eval_dataset = CausalPredictionDataset(
        validation[validation.user_id.isin(validation_users)],
        max_length=MAX_LENGTH,
        relevance_col=RELEVANCE_COL,
        relevance_threshold=RELEVANCE_THRESHOLD,
        user_col=USER_COL,
        validation_mode=True,
        positive_eval=only_positive,
    )

    return eval_dataset


train_dataset = BarlowDataset(
    train,
    user_col=USER_COL,
    max_length=MAX_LENGTH,
    relevance_col=RELEVANCE_COL,
    relevance_threshold=RELEVANCE_THRESHOLD,
)
eval_dataset = get_eval_dataset(validation)

collate_fn_train = PaddingCollateFn(add_aug_mask=True, labels_keys=['labels', 'aug_labels'])
collate_fn_val = PaddingCollateFn()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_train,
    persistent_workers=True
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_val,
    persistent_workers=True
)

predict_pos_dataset = CausalPredictionDataset(
    test_pos,
    user_col=USER_COL,
    max_length=MAX_LENGTH,
    relevance_col=RELEVANCE_COL,
    relevance_threshold=RELEVANCE_THRESHOLD,
    positive_eval=only_positive,
)
predict_neg_dataset = CausalPredictionDataset(
    test_neg,
    user_col=USER_COL,
    max_length=MAX_LENGTH,
    relevance_col=RELEVANCE_COL,
    relevance_threshold=RELEVANCE_THRESHOLD,
    positive_eval=only_positive,
)
predict_pos_loader = DataLoader(
    predict_pos_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_val,
    persistent_workers=True
)
predict_neg_loader = DataLoader(
    predict_neg_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_val,
    persistent_workers=True
)

item_count = train.item_id.max()
add_head = True

def main(cfg):
    logger = CSVLogger("", name="metrics.csv")
    fix_seed(SEED)
    model = SASRecBarlow(
        item_num=item_count,
        add_head=add_head,
        maxlen=MAX_LENGTH,
        num_heads=1,
        dropout_rate=dropout,
        hidden_units=hidden_units,
        num_blocks=num_blocks,
    )

    fix_seed(SEED)
    seqrec_module = SeqRecBarlow(model,
                                 lr=lr,
                                 predict_top_k=10,
                                 filter_seen=True,
                                 barlow_coeff=barlow_coeff, 
                                 off_diag_coeff=off_diag_coeff, 
                                 power_coef=power_coef)
    early_stopping = EarlyStopping(
        monitor="val_ndcg", mode="max", patience=10, verbose=False
    )

    model_summary = ModelSummary(max_depth=2)
    checkpoint = ModelCheckpoint(
        save_top_k=1, monitor="val_ndcg", mode="max", save_weights_only=True
    )
    callbacks = [early_stopping, model_summary, checkpoint]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        accelerator="auto",  # Specify the accelerator type
        devices="auto",
        max_epochs=100,
        deterministic=True,
    )

    trainer.fit(
        model=seqrec_module,
        train_dataloaders=train_loader,
        val_dataloaders=eval_loader,
    )

    seqrec_module.load_state_dict(
        torch.load(checkpoint.best_model_path)["state_dict"]
    )
    history = pd.read_csv(os.path.join(trainer.logger.experiment.log_dir, 'metrics.csv'))
    val_metrics = {
        "val_ndcg": history["val_ndcg"].max(),
        "val_hit_rate": history["val_hit_rate"].max(),
        "val_mrr": history["val_mrr"].max(),
    }

    seqrec_module.predict_top_k = 10
    seqrec_module.filter_seen = True
    preds_pos = trainer.predict(model=seqrec_module, dataloaders=predict_pos_loader)
    preds_neg = trainer.predict(model=seqrec_module, dataloaders=predict_neg_loader)
    recs_pos = preds2recs(preds_pos)
    recs_neg = preds2recs(preds_neg)

    metrics = compute_metrics(
        last_item_pos,
        last_item_neg,
        recs_pos,
        recs_neg,
        train_full,
        relevance_col=RELEVANCE_COL,
        relevance_threshold=RELEVANCE_THRESHOLD,
    )
    
    print(metrics)


if __name__ == "__main__":

    main(cfg)
