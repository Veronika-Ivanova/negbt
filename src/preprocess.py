"""
Filter interactions.
"""
import os

import pandas as pd
from tqdm import tqdm


def train_val_test_split(
    df,
    relevance_threshold,
    relevance_col,
    train_quantile=0.8,
    val_quantile=0.9,
    item_min_count=5,
    user_min_count=5,
    min_items_per_user=2,
    user_col='user_id',
    item_col='item_id',
    timestamp_col='timestamp',
):
    """
    Disjoint split by global timestamp (default 80% / 10% / 10%).

    - train: ``timestamp < train_timepoint``
    - validation window: ``train_timepoint <= timestamp < val_timepoint``
    - test window: ``timestamp >= val_timepoint``
    """
    df = df.sort_values([user_col, timestamp_col])
    df[user_col] = df[user_col].astype('category').cat.codes
    df[item_col] = df[item_col].astype('category').cat.codes
    df = filter_items(df, item_min_count, item_col=item_col)
    df = filter_users(df, user_min_count, user_col=user_col)

    train_timepoint = df[timestamp_col].quantile(
        q=train_quantile, interpolation='nearest'
    )
    val_timepoint = df[timestamp_col].quantile(
        q=val_quantile, interpolation='nearest'
    )

    train = df.query('timestamp < @train_timepoint')
    val = df.query('@train_timepoint <= timestamp < @val_timepoint')
    test = df.query('@val_timepoint <= timestamp')

    train = filter_users_by_history_len(
        train, user_col, relevance_col, relevance_threshold, min_items_per_user
    )

    train_users = train[user_col].unique()
    train_items = train[item_col].unique()
    val = val[val[user_col].isin(train_users) & val[item_col].isin(train_items)]
    test = test[test[user_col].isin(train_users) & test[item_col].isin(train_items)]

    val = filter_users_with_pos_neg(val, relevance_col, relevance_threshold, user_col)
    test = filter_users_with_pos_neg(test, relevance_col, relevance_threshold, user_col)

    train = add_time_idx(train, user_col=user_col, timestamp_col=timestamp_col)
    val = add_time_idx(val, user_col=user_col, timestamp_col=timestamp_col)
    test = add_time_idx(test, user_col=user_col, timestamp_col=timestamp_col)

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    return train, val, test


def add_time_idx(df, user_col='user_id', timestamp_col='timestamp', sort=True):
    """Add time index to interactions dataframe."""

    if sort:
        df = df.sort_values([user_col, timestamp_col])

    df['time_idx'] = df.groupby(user_col).cumcount()
    df['time_idx_reversed'] = df.groupby(user_col).cumcount(ascending=False)

    return df


def extract_last_neighbour_pos_neg_pair(
    df,
    relevance_col,
    relevance_threshold,
    user_col='user_id',
    time_col='time_idx',
    target_window_df=None,
):
    """
    Per user, find the last adjacent positive/negative pair in chronological order.

    If ``target_window_df`` is given, only pairs whose both interactions belong to that
    window are eligible as targets. The pair and all later interactions are removed
    from ``df`` (full history).
    """
    df = df.sort_values([user_col, time_col]).copy()
    target_indices = (
        set(target_window_df.index)
        if target_window_df is not None
        else set(df.index)
    )

    last_pos_rows = []
    last_neg_rows = []
    drop_indices = []

    for _, group in df.groupby(user_col, sort=False):
        relevances = group[relevance_col].to_numpy()
        is_pos = relevances >= relevance_threshold
        last_i = None
        for i in range(len(is_pos) - 1):
            if is_pos[i] == is_pos[i + 1]:
                continue
            idx_i = group.index[i]
            idx_i1 = group.index[i + 1]
            if idx_i in target_indices and idx_i1 in target_indices:
                last_i = i
        if last_i is None:
            continue

        if is_pos[last_i]:
            pos_row, neg_row = group.iloc[last_i], group.iloc[last_i + 1]
        else:
            neg_row, pos_row = group.iloc[last_i], group.iloc[last_i + 1]

        last_pos_rows.append(pos_row)
        last_neg_rows.append(neg_row)
        drop_indices.extend(group.iloc[last_i:].index.tolist())

    remaining = df.drop(index=drop_indices)
    last_pos = pd.DataFrame(last_pos_rows).reset_index(drop=True)
    last_neg = pd.DataFrame(last_neg_rows).reset_index(drop=True)
    if len(last_pos) > 0:
        valid_users = last_pos[user_col].unique()
        remaining = remaining[remaining[user_col].isin(valid_users)]
    else:
        remaining = remaining.iloc[0:0]
    return remaining, last_pos, last_neg


def restrict_to_train_catalog(df, train, user_col='user_id', item_col='item_id'):
    """Drop cold users and items relative to train."""
    train_users = train[user_col].unique()
    train_items = train[item_col].unique()
    return df[
        df[user_col].isin(train_users) & df[item_col].isin(train_items)
    ]


def verify_split_integrity(
    train,
    validation,
    test,
    last_pos_item_val,
    last_neg_item_val,
    last_pos_item_test,
    last_neg_item_test,
    user_col='user_id',
    item_col='item_id',
    timestamp_col='timestamp',
):
    """Raise ``ValueError`` if train/validation/test leak targets or cold entities."""
    train_users = set(train[user_col].unique())
    train_items = set(train[item_col].unique())
    val_users = set(validation[user_col].unique())
    test_users = set(test[user_col].unique())
    val_items = set(validation[item_col].unique())
    test_items = set(test[item_col].unique())

    if not val_users.issubset(train_users):
        raise ValueError('Validation has cold users not present in train.')
    if not test_users.issubset(train_users):
        raise ValueError('Test has cold users not present in train.')
    if not val_items.issubset(train_items):
        raise ValueError('Validation has cold items not present in train.')
    if not test_items.issubset(train_items):
        raise ValueError('Test has cold items not present in train.')

    def interaction_keys(frame):
        return set(zip(frame[user_col], frame[item_col], frame[timestamp_col]))

    train_keys = interaction_keys(train)
    val_keys = interaction_keys(validation)
    val_history_keys = interaction_keys(validation)
    test_history_keys = interaction_keys(test)

    for name, targets in (
        ('validation', last_pos_item_val),
        ('validation', last_neg_item_val),
    ):
        for row in targets.itertuples():
            key = (getattr(row, user_col), getattr(row, item_col), getattr(row, timestamp_col))
            if key in val_history_keys:
                raise ValueError(f'{name} target interaction is still in {name} history: {key}')
            if key in train_keys:
                raise ValueError(f'{name} target interaction is present in train: {key}')

    for name, targets in (
        ('test', last_pos_item_test),
        ('test', last_neg_item_test),
    ):
        for row in targets.itertuples():
            key = (getattr(row, user_col), getattr(row, item_col), getattr(row, timestamp_col))
            if key in test_history_keys:
                raise ValueError(f'{name} target interaction is still in {name} history: {key}')
            if key in val_keys:
                raise ValueError(f'{name} target interaction is present in validation history: {key}')
            if key in train_keys:
                raise ValueError(f'{name} target interaction is present in train: {key}')


def prepare_splitted_data(data_path,
                          relevance_col,
                          relevance_threshold,
                          user_col='user_id',
                          item_col='item_id',
                          min_items_per_user=2,
                          timestamp_col='timestamp',
                          filter_negative=False,
                          verify=True):
    """
    Load train / validation / test parquets produced by ``train_val_test_split``.

    Builds cumulative histories: validation = train + val window;
    test = train + val window + test window (per user).
    Targets are taken only from the val/test windows.
    """
    train = pd.read_parquet(os.path.join(data_path, 'train.parquet'))
    validation = pd.read_parquet(os.path.join(data_path, 'validation.parquet'))
    test = pd.read_parquet(os.path.join(data_path, 'test.parquet'))

    for frame in (train, validation, test):
        frame[item_col] = frame[item_col] + 1

    val_window = validation.copy()
    test_window = test.copy()

    if filter_negative:
        train = train[train[relevance_col] >= relevance_threshold]

    train = restrict_to_train_catalog(train, train, user_col, item_col)
    val_window = restrict_to_train_catalog(val_window, train, user_col, item_col)
    test_window = restrict_to_train_catalog(test_window, train, user_col, item_col)

    train = train.copy()
    train['_split_part'] = 'train'
    val_window = val_window.copy()
    val_window['_split_part'] = 'val'
    test_window = test_window.copy()
    test_window['_split_part'] = 'test'

    train_for_val = train[train[user_col].isin(val_window[user_col].unique())]
    validation = pd.concat([train_for_val, val_window], ignore_index=True)

    train_for_test = train[train[user_col].isin(test_window[user_col].unique())]
    val_for_test = val_window[val_window[user_col].isin(test_window[user_col].unique())]
    test = pd.concat([train_for_test, val_for_test, test_window], ignore_index=True)

    validation = restrict_to_train_catalog(validation, train, user_col, item_col)
    test = restrict_to_train_catalog(test, train, user_col, item_col)

    train = add_time_idx(train.drop(columns=['_split_part']), user_col=user_col, timestamp_col=timestamp_col)
    validation = add_time_idx(validation, user_col=user_col, timestamp_col=timestamp_col)
    test = add_time_idx(test, user_col=user_col, timestamp_col=timestamp_col)

    val_target_window = validation[validation['_split_part'] == 'val'].drop(columns=['_split_part'])
    test_target_window = test[test['_split_part'] == 'test'].drop(columns=['_split_part'])
    validation = validation.drop(columns=['_split_part'])
    test = test.drop(columns=['_split_part'])

    validation, last_pos_item_val, last_neg_item_val = extract_last_neighbour_pos_neg_pair(
        validation,
        relevance_col,
        relevance_threshold,
        user_col=user_col,
        target_window_df=val_target_window,
    )
    test, last_pos_item_test, last_neg_item_test = extract_last_neighbour_pos_neg_pair(
        test,
        relevance_col,
        relevance_threshold,
        user_col=user_col,
        target_window_df=test_target_window,
    )

    if verify:
        verify_split_integrity(
            train,
            validation,
            test,
            last_pos_item_val,
            last_neg_item_val,
            last_pos_item_test,
            last_neg_item_test,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
        )

    return (
        train,
        validation,
        test,
        last_pos_item_test,
        last_pos_item_val,
        last_neg_item_test,
        last_neg_item_val,
    )


def filter_users_with_pos_neg(df, relevance_col, relevance_threshold, user_col='user_id'):
    """Keep users that have at least one positive and one negative interaction."""
    pos_mask = df[relevance_col] >= relevance_threshold
    neg_mask = df[relevance_col] < relevance_threshold
    pos_users = df.loc[pos_mask, user_col].unique()
    neg_users = df.loc[neg_mask, user_col].unique()
    valid_users = set(pos_users) & set(neg_users)
    return df[df[user_col].isin(valid_users)]


def filter_users_by_history_len(df,
                                user_col,
                                relevance_col,
                                relevance_threshold,
                                min_items_per_user=2,
                                filter_by_positive_items=True):
    if filter_by_positive_items:
        user_count = df[df[relevance_col] >= relevance_threshold][user_col].value_counts()
    else:
        user_count = df[user_col].value_counts()
    appropriate_users = user_count[user_count >= min_items_per_user].index
    df = df[df.loc[:, user_col].isin(appropriate_users)]
    return df


def filter_items(df, item_min_count, item_col='item_id'):

    print('Filtering items..')

    item_count = df.groupby(item_col).user_id.nunique()

    item_ids = item_count[item_count >= item_min_count].index
    print(f'Number of items before {len(item_count)}')
    print(f'Number of items after {len(item_ids)}')

    print(f'Interactions length before: {len(df)}')
    df = df[df[item_col].isin(item_ids)]
    print(f'Interactions length after: {len(df)}')

    return df


def filter_users(df, user_min_count, user_col='user_id'):

    print('Filtering users..')

    user_count = df.groupby(user_col).item_id.nunique()

    user_ids = user_count[user_count >= user_min_count].index
    print(f'Number of users before {len(user_count)}')
    print(f'Number of users after {len(user_ids)}')

    print(f'Interactions length before: {len(df)}')
    df = df[df[user_col].isin(user_ids)]
    print(f'Interactions length after: {len(df)}')

    return df
