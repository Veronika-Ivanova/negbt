"""
Torch datasets and collate function.
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CausalDataset(Dataset):
    """Training dataset for negative feedback."""

    padding_value = 0
    labels_padding_value = -100

    def __init__(self, df, max_length,
                 relevance_col, relevance_threshold,
                 user_col, item_col='item_id',
                 time_col='time_idx'):
        self.max_length = max_length
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.relevance_col = relevance_col
        self.relevance_threshold = relevance_threshold
        self._prepare_data(df)

    def _prepare_data(self, df):
        df = df.sort_values(self.time_col)
        self.items = df.groupby(self.user_col)[self.item_col].agg(list).to_dict()
        self.relevances = df.groupby(self.user_col)[self.relevance_col].agg(list).to_dict()
        self.user_ids = list(self.items.keys())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_sequence = self.items[user_id]
        relevance_sequence = self.relevances[user_id]

        item_sequence = np.array(item_sequence).astype(float)
        item_sequence = item_sequence[-self.max_length - 1:]
        relevance_sequence = relevance_sequence[-self.max_length - 1:]

        input_ids = item_sequence[:-1].astype(int)
        labels = item_sequence[1:].astype(int)
        skip = (np.array(relevance_sequence)[:-1] < self.relevance_threshold).astype(int)

        item = {'user_id': user_id, 'input_ids': input_ids,
                'labels': labels, 'skip': skip}

        return item


class CausalPredictionDataset(CausalDataset):
    """Prediction / validation dataset for negative feedback."""

    def __init__(self, df, max_length,
                 relevance_col, relevance_threshold,
                 user_col, item_col='item_id',
                 time_col='time_idx',
                 validation_mode=False, validation_size=None, positive_eval=True,
                 last_pos_targets=None, last_neg_targets=None):
        self.max_length = max_length
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.relevance_col = relevance_col
        self.relevance_threshold = relevance_threshold
        self.validation_mode = validation_mode
        self.validation_size = validation_size
        self.positive_eval = positive_eval
        self.target_pos = None
        self.target_neg = None

        if validation_mode:
            if last_pos_targets is None or last_neg_targets is None:
                raise ValueError(
                    'last_pos_targets and last_neg_targets are required in validation_mode '
                    '(e.g. last_pos_item_val and last_neg_item_val from prepare_splitted_data).'
                )
            self.target_pos, self.target_neg, users = self._prepare_targets(
                last_pos_targets, last_neg_targets, user_col, item_col
            )
            df = df[df[user_col].isin(users)]
            users = np.array(list(users))
        else:
            users = df[user_col].unique()

        if validation_size and (validation_size < len(users)):
            users = np.random.choice(users, size=validation_size, replace=False)
            df = df[df[user_col].isin(users)]

        self._prepare_data(df)
        users_set = set(users)
        self.user_ids = [u for u in self.user_ids if u in users_set]

    def _prepare_targets(self, last_pos, last_neg, user_col, item_col):
        pos = last_pos.set_index(user_col)[item_col].to_dict()
        neg = last_neg.set_index(user_col)[item_col].to_dict()
        users = set(pos.keys()) & set(neg.keys())
        return pos, neg, users

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_sequence = np.array(self.items[user_id]).astype(float)
        relevance_sequence = np.array(self.relevances[user_id])

        item = {
            'user_id': user_id,
            'full_history': item_sequence.astype(int),
        }

        if self.validation_mode:
            item['target_pos'] = int(self.target_pos[user_id])
            item['target_neg'] = int(self.target_neg[user_id])

        if self.positive_eval:
            positive_input_ids = item_sequence[
                relevance_sequence >= self.relevance_threshold
            ].astype(int)
            item['input_ids'] = positive_input_ids[-self.max_length:]
        else:
            item['input_ids'] = item['full_history'][-self.max_length:]

        return item


class PaddingCollateFn:
    """Collate function for aggregating in batch."""

    def __init__(self, padding_value=0,
                 labels_padding_value=-100, labels_keys=['labels'],
                 add_aug_mask=False, add_negative_mask=False, bert=False):

        self.padding_value = padding_value
        self.labels_padding_value = labels_padding_value
        self.labels_keys = labels_keys
        self.add_negative_mask = add_negative_mask
        self.add_aug_mask = add_aug_mask
        self.bert = bert

    def __call__(self, batch):
        collated_batch = {}

        for key in batch[0].keys():
            if np.isscalar(batch[0][key]):
                collated_batch[key] = torch.tensor([example[key] for example in batch])
                continue

            if key in self.labels_keys:
                padding_value = self.labels_padding_value
            else:
                padding_value = self.padding_value

            values = [torch.tensor(example[key]) for example in batch]
            collated_batch[key] = pad_sequence(values, batch_first=True,
                                               padding_value=padding_value)

        positive_attention_mask = collated_batch['input_ids'] != self.padding_value
        collated_batch['attention_mask'] = positive_attention_mask.to(dtype=torch.float32)

        if self.add_aug_mask:
            aug_attention_mask = collated_batch['aug_input_ids'] != self.padding_value
            collated_batch['aug_attention_mask'] = aug_attention_mask.to(dtype=torch.float32)

        if self.add_negative_mask:
            negative_attention_mask = collated_batch['negative_input_ids'] != self.padding_value
            collated_batch['negative_attention_mask'] = negative_attention_mask.to(dtype=torch.float32)

        if self.bert:
            attention_mask = collated_batch['input_ids'] != self.padding_value
            collated_batch['attention_mask'] = attention_mask.to(dtype=torch.float32)

        return collated_batch


class BarlowDataset(Dataset):
    """Training dataset for NegBT."""

    padding_value = 0
    labels_padding_value = -100

    def __init__(self, df, max_length,
                 relevance_col, relevance_threshold,
                 user_col, item_col='item_id',
                 time_col='time_idx'):
        self.max_length = max_length
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.relevance_col = relevance_col
        self.relevance_threshold = relevance_threshold
        self._prepare_data(df)

    def _prepare_data(self, df):
        df = df.sort_values(self.time_col)
        self.items = df.groupby(self.user_col)[self.item_col].agg(list).to_dict()
        self.relevances = df.groupby(self.user_col)[self.relevance_col].agg(list).to_dict()
        self.user_ids = list(self.items.keys())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_sequence = self.items[user_id]
        relevance_sequence = self.relevances[user_id]
        item_sequence = np.array(item_sequence).astype(float)

        positive_item_sequence = item_sequence[np.array(relevance_sequence) >= self.relevance_threshold]
        positive_item_sequence = positive_item_sequence[-self.max_length - 1:]
        positive_input_ids = positive_item_sequence[:-1].astype(int)
        positive_labels = positive_item_sequence[1:].astype(int)

        last_positive_index = np.where(item_sequence == positive_input_ids[-1])[-1][-1]
        input_ids = item_sequence[:last_positive_index + 1][-self.max_length:].astype(int)

        item = {'user_id': user_id, 'input_ids': positive_input_ids, 'aug_input_ids': input_ids,
                'labels': positive_labels}

        return item


class BTSRDataset(Dataset):
    """
    BT-SR training dataset (barlow_twins_sasrec).

    Full interaction sequence for CE; semantic augmentation uses another user's
    prefix when the last item (target) matches.
    """

    padding_value = 0
    labels_padding_value = -100

    def __init__(self, df, max_length,
                 user_col, item_col='item_id',
                 time_col='time_idx', seed=42):
        self.max_length = max_length
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.rng = np.random.RandomState(seed)
        self._prepare_data(df)

    def _prepare_data(self, df):
        df = df.sort_values(self.time_col)
        self.items = df.groupby(self.user_col)[self.item_col].agg(list).to_dict()
        self.user_ids = [
            uid for uid, seq in self.items.items() if len(seq) >= 2
        ]
        self.same_target_users = {}
        target_to_users = {}
        for uid in self.user_ids:
            target = self.items[uid][-1]
            target_to_users.setdefault(target, []).append(uid)
        for uid in self.user_ids:
            target = self.items[uid][-1]
            self.same_target_users[uid] = [
                u for u in target_to_users[target] if u != uid
            ]

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        user_items = np.array(self.items[user_id]).astype(int)
        user_items = user_items[-self.max_length - 1:]
        input_ids = user_items[:-1]
        labels = user_items[1:]

        aug_users = self.same_target_users[user_id]
        if aug_users:
            aug_uid = self.rng.choice(aug_users)
        else:
            aug_uid = user_id
        aug_items = np.array(self.items[aug_uid]).astype(int)
        aug_items = aug_items[-self.max_length - 1:]
        aug_input_ids = aug_items[:-1]

        return {
            'user_id': user_id,
            'input_ids': input_ids,
            'labels': labels,
            'aug_input_ids': aug_input_ids,
        }


class CausalNegativeFeedbackDataset(Dataset):
    """Training dataset with separate positive and negative sequences."""

    padding_value = 0
    labels_padding_value = -100

    def __init__(self, df, max_length,
                 relevance_col, relevance_threshold,
                 user_col, item_col='item_id',
                 time_col='time_idx'):
        self.max_length = max_length
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.relevance_col = relevance_col
        self.relevance_threshold = relevance_threshold
        self._prepare_data(df)

    def _prepare_data(self, df):
        df = df.sort_values(self.time_col)
        self.items = df.groupby(self.user_col)[self.item_col].agg(list).to_dict()
        self.relevances = df.groupby(self.user_col)[self.relevance_col].agg(list).to_dict()
        self.user_ids = list(self.items.keys())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_sequence = self.items[user_id]
        relevance_sequence = self.relevances[user_id]

        item_sequence = np.array(item_sequence).astype(float)

        positive_item_sequence = item_sequence[np.array(relevance_sequence) >= self.relevance_threshold]
        negative_item_sequence = item_sequence[np.array(relevance_sequence) < self.relevance_threshold]

        positive_item_sequence = positive_item_sequence[-self.max_length - 1:]
        negative_item_sequence = negative_item_sequence[-self.max_length - 1:]

        positive_input_ids = positive_item_sequence[:-1].astype(int)
        negative_input_ids = negative_item_sequence[:-1].astype(int)
        positive_labels = positive_item_sequence[1:].astype(int)
        negative_labels = negative_item_sequence[1:].astype(int)

        item = {'user_id': user_id, 'input_ids': positive_input_ids, 'negative_input_ids': negative_input_ids,
                'labels': positive_labels, 'negative_labels': negative_labels}

        return item
