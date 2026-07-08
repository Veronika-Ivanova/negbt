"""
Train BT-SR on the same NegBT splits and metrics.

Run from negbt/src:
    python ../runs/run_btsr.py
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

import pandas as pd
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader

from datasets import PaddingCollateFn, BTSRDataset, CausalPredictionDataset
from modules import SeqRecBtSr
from models import SASRecBarlow
from postprocess import preds2recs
from preprocess import prepare_splitted_data
from metrics import compute_metrics
from utils import fix_seed, get_csv_metrics_path

cfg = OmegaConf.load("../runs/configs/BTSR_ml.yaml")

PROJECT_PATH = f"{cfg.project_path}"
DATA_PATH = f"{PROJECT_PATH}{cfg.data_path}"

USER_COL = f"{cfg.dataset.user_col}"
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

fix_seed(SEED)
(
    train,
    validation,
    test,
    last_pos_item_test,
    last_pos_item_val,
    last_neg_item_test,
    last_neg_item_val,
) = prepare_splitted_data(
    DATA_PATH,
    user_col=USER_COL,
    relevance_col=RELEVANCE_COL,
    filter_negative=False,
    relevance_threshold=RELEVANCE_THRESHOLD,
)


def get_eval_dataset(
    validation,
    last_pos_item_val,
    last_neg_item_val,
    validation_size=VALIDATION_SIZE,
):
    return CausalPredictionDataset(
        validation,
        max_length=MAX_LENGTH,
        relevance_col=RELEVANCE_COL,
        relevance_threshold=RELEVANCE_THRESHOLD,
        user_col=USER_COL,
        validation_mode=True,
        positive_eval=only_positive,
        last_pos_targets=last_pos_item_val,
        last_neg_targets=last_neg_item_val,
        validation_size=validation_size,
    )


train_dataset = BTSRDataset(
    train,
    user_col=USER_COL,
    max_length=MAX_LENGTH,
    seed=SEED,
)
eval_dataset = get_eval_dataset(validation, last_pos_item_val, last_neg_item_val)

collate_fn_train = PaddingCollateFn(add_aug_mask=True, labels_keys=['labels'])
collate_fn_val = PaddingCollateFn()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_train,
    persistent_workers=True,
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_val,
    persistent_workers=True,
)

predict_test_dataset = CausalPredictionDataset(
    test,
    user_col=USER_COL,
    max_length=MAX_LENGTH,
    relevance_col=RELEVANCE_COL,
    relevance_threshold=RELEVANCE_THRESHOLD,
    positive_eval=only_positive,
)
predict_test_loader = DataLoader(
    predict_test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_val,
    persistent_workers=True,
)

item_count = train.item_id.max()
add_head = True


def main(cfg):
    logger = CSVLogger("", name="metrics_btsr")
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

    seqrec_module = SeqRecBtSr(
        model,
        lr=lr,
        predict_top_k=10,
        filter_seen=True,
        barlow_coeff=barlow_coeff,
        off_diag_coeff=off_diag_coeff,
        power_coef=1.0,
    )

    early_stopping = EarlyStopping(
        monitor="val_ndcg_pos", mode="max", patience=cfg.patience, verbose=False
    )
    model_summary = ModelSummary(max_depth=2)
    checkpoint = ModelCheckpoint(
        save_top_k=1, monitor="val_ndcg_pos", mode="max", save_weights_only=True
    )
    callbacks = [early_stopping, model_summary, checkpoint]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        accelerator=cfg.trainer_params.get("accelerator", "auto"),
        devices=cfg.trainer_params.get("devices", 1),
        strategy=cfg.trainer_params.get("strategy", "auto"),
        max_epochs=cfg.trainer_params.max_epochs,
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
    history = pd.read_csv(get_csv_metrics_path(trainer))
    print({
        "val_ndcg_pos": history["val_ndcg_pos"].max(),
        "val_hit_rate_pos": history["val_hit_rate_pos"].max(),
        "val_mrr_pos": history["val_mrr_pos"].max(),
    })

    preds_test = trainer.predict(model=seqrec_module, dataloaders=predict_test_loader)
    recs_test = preds2recs(preds_test)
    metrics = compute_metrics(
        last_pos_item_test,
        last_neg_item_test,
        recs_test,
        train,
    )
    print(metrics)


if __name__ == "__main__":
    main(cfg)
