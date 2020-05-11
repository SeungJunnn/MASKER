import os
import random

import torch
from torch.utils.data import TensorDataset


class MaskedDataset(object):
    def __init__(self, base_dataset, keyword_type, keyword_num):
        self.base_dataset = base_dataset
        self.keyword_type = keyword_type
        if self.keyword_type in ['tfidf', 'attention']:
            self.keyword_num = keyword_num  # specify number of keywords
        else:
            self.keyword_num = len(base_dataset.tokenizer)  # use full tokens

        if not self._check_exists():
            self._preprocess()

        self.train_dataset = torch.load(self._train_path)  # masked dataset
        self.test_dataset = base_dataset.test_dataset
        self.tokenizer = base_dataset.tokenizer
        self.n_classes = base_dataset.n_classes

    @property
    def _train_path(self):
        train_path = self.base_dataset._train_path
        train_path = train_path.replace('train', 'train_{}_{}'.format(self.keyword_type, self.keyword_num))
        return train_path

    def _check_exists(self):
        if os.path.exists(self._train_path):
            return True
        else:
            return False

    def _preprocess(self):
        tokenizer = self.base_dataset.tokenizer
        dataset = self.base_dataset.train_dataset
        seed = self.base_dataset.seed

        if self.keyword_type in ['tfidf', 'attention']:
            if self.keyword_type == 'tfidf':
                keyword = get_tfidf_keyword()
            else:
                keyword = get_attention_keyword()

            masked_dataset = get_masked_dataset(tokenizer, dataset, keyword=keyword,
                                                seed=seed, key_mask_ratio=0.5)
        elif self.keyword_type == 'random':
            masked_dataset = get_masked_dataset(tokenizer, dataset,
                                                seed=seed, key_mask_ratio=0.15)
        else:
            raise ValueError('No matching keyword type')

        torch.save(masked_dataset, self._train_path)


def get_tfidf_keyword():
    raise NotImplementedError


def get_attention_keyword():
    raise NotImplementedError


def get_masked_dataset(tokenizer, dataset, keyword=None,
                       seed=0, key_mask_ratio=0.5, out_mask_ratio=0.9):

    CLS_TOKEN = tokenizer.cls_token_id
    PAD_TOKEN = tokenizer.pad_token_id
    MASK_TOKEN = tokenizer.mask_token_id

    random.seed(seed)  # fix random seed

    tokens = dataset.tensors[0]
    labels = dataset.tensors[1]

    masked_tokens = []
    masked_labels = []

    for (token, label) in zip(tokens, labels):
        m_token = token.clone()  # masked token (for self-supervision)
        o_token = token.clone()  # outlier token (for entropy regularization)
        m_label = -torch.ones(token.size(0) + 1).long()  # self-sup labels + class label

        for i, tok in enumerate(token):
            if tok == CLS_TOKEN:
                continue
            elif tok == PAD_TOKEN:
                break

            if (keyword is None) or (tok in keyword):
                if random.random() < key_mask_ratio:  # randomly mask keywords
                    m_token[i] = MASK_TOKEN
                    if keyword is None:
                        m_label[i] = tok  # use full vocabulary
                    else:
                        m_label[i] = keyword.index(tok)  # convert to keyword index
            else:
                if random.random() < out_mask_ratio:  # randomly mask non-keywords
                    o_token[i] = MASK_TOKEN

        m_label[-1] = label  # class label

        masked_tokens.append(torch.cat([token, m_token, o_token]))  # (original, masked, outlier)
        masked_labels.append(m_label)

    masked_tokens = torch.stack(masked_tokens)
    masked_labels = torch.stack(masked_labels)

    masked_dataset = TensorDataset(masked_tokens, masked_labels)

    return masked_dataset


