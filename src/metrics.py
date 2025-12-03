"""
Metrics.
"""
import numpy as np
import pandas as pd
import torch
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, recall_at_k
from tqdm.auto import tqdm


def compute_metrics(last_item_pos,
    last_item_neg,
    preds_pos,
    preds_neg, train_full, relevance_col,
                    relevance_threshold, k=10):
    
    # when we have 1 true positive, HitRate == Recall and MRR == MAP
    metrics_dict = {
        f'NDCG_p': round(ndcg_at_k(last_item_pos, preds_pos, col_user='user_id', col_item='item_id',
                             col_prediction='prediction', col_rating=relevance_col, k=k), 6),
        f'HR_p': round(recall_at_k(last_item_pos, preds_pos, col_user='user_id', col_item='item_id',
                             col_prediction='prediction', col_rating=relevance_col, k=k), 6),
        f'MRR_p': round(map_at_k(last_item_pos, preds_pos, col_user='user_id', col_item='item_id',
                           col_prediction='prediction', col_rating=relevance_col, k=k), 6),
        f'NDCG_n': round(ndcg_at_k(last_item_neg, preds_neg, col_user='user_id', col_item='item_id',
                             col_prediction='prediction', col_rating=relevance_col, k=k), 6),
        f'HR_n': round(recall_at_k(last_item_neg, preds_neg, col_user='user_id', col_item='item_id',
                             col_prediction='prediction', col_rating=relevance_col, k=k), 6),
        f'MRR_n': round(map_at_k(last_item_neg, preds_neg, col_user='user_id', col_item='item_id',
                           col_prediction='prediction', col_rating=relevance_col, k=k), 6),
        f'Coverage': round(pd.concat([preds_pos, preds_neg]).item_id.nunique() / train_full.item_id.nunique(), 6),
    }
    
    metrics_dict['HR_diff'] = round(metrics_dict['HR_p'] - metrics_dict['HR_n'], 6)
    metrics_dict['MRR_diff'] = round(metrics_dict['MRR_p'] - metrics_dict['MRR_n'], 6)
    metrics_dict['NDCG_diff'] = round(metrics_dict['NDCG_p'] - metrics_dict['NDCG_n'], 6)

    return metrics_dict
