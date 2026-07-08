"""
Metrics.
"""
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def compute_metrics(last_pos_item_test,
                    last_neg_item_test,
                    recs,
                    train,
                    user_col='user_id',
                    item_col='item_id'):
    recs_list = (recs
                 .rename(columns={'user_id': user_col})
                 .groupby(user_col)[item_col]
                 .agg(lambda x: list(x))
                 .reset_index()
                 .rename(columns={item_col: 'item_ids'}))
    
    metrics_dict = {
                    'HR_p': hr(last_pos_item_test, recs_list, user_col, item_col),
                    'MRR_p': mrr(last_pos_item_test, recs_list, user_col, item_col),
                    'NDCG_p': ndcg(last_pos_item_test, recs_list, user_col, item_col),
                    'HR_n': hr(last_neg_item_test, recs_list, user_col, item_col),
                    'MRR_n': mrr(last_neg_item_test, recs_list, user_col, item_col),
                    'NDCG_n': ndcg(last_neg_item_test, recs_list, user_col, item_col)}
    
    metrics_dict['coverage'] = coverage(recs, train, item_col)
    return metrics_dict

def hr(
    ground_truth: pd.DataFrame,
    recs_list: pd.DataFrame,
    user_col='user_id',
    item_col='item_id',
) -> float:
    df = ground_truth.merge(recs_list, on=user_col, how='inner')
    hr_values = []
    for _, row in df.iterrows():
        hr_values.append(int(row[item_col] in row['item_ids']))
    return round(np.mean(hr_values), 6)

def mrr(
    ground_truth: pd.DataFrame,
    recs_list: pd.DataFrame,
    user_col='user_id',
    item_col='item_id'
) -> float:
    df = ground_truth.merge(recs_list, on=user_col, how='inner')
    mrr_values = []
    for _, row in df.iterrows():
        try:
            user_mrr = 1 / (row['item_ids'].index(row[item_col]) + 1)
        except ValueError:
            user_mrr = 0
        mrr_values.append(user_mrr)
    return round(np.mean(mrr_values), 6)

def ndcg(
    ground_truth: pd.DataFrame,
    recs_list: pd.DataFrame,
    user_col='user_id',
    item_col='item_id'
) -> float:
    # ideal dcg == 1 при стратегии разделения leave-one-out
    df = ground_truth.merge(recs_list, on=user_col, how='inner')
    ndcg_values = []
    for _, row in df.iterrows():
        try:
            user_ndcg = 1 / np.log2(row['item_ids'].index(row[item_col]) + 2)
        except ValueError:
            user_ndcg = 0
        ndcg_values.append(user_ndcg)
    return round(np.mean(ndcg_values), 6)

def coverage(recs: pd.DataFrame, train: pd.DataFrame, item_col='item_id') -> float:
    return round(recs[item_col].nunique() / train[item_col].nunique(), 6)
