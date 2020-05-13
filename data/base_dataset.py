import os
import json
from abc import *

import torch
from torch.utils.data import TensorDataset
import numpy as np

from common import DATA_PATH


def tokenize(tokenizer, raw_text, max_len=512):
    if len(raw_text) > max_len:
        raw_text = raw_text[:max_len]

    tokens = tokenizer.encode(raw_text, add_special_tokens=True)
    tokens = torch.tensor(tokens).long()

    if tokens.size(0) < max_len:
        padding = torch.zeros(max_len - tokens.size(0)).long()
        padding.fill_(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
        tokens = torch.cat([tokens, padding])

    return tokens


def create_tensor_dataset(tokens, labels):
    assert len(tokens) == len(labels)

    tokens = torch.stack(tokens)  # (N, T)
    labels = torch.stack(labels).unsqueeze(1)  # (N, 1)

    dataset = TensorDataset(tokens, labels)

    return dataset


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, data_name, total_class, tokenizer,
                 max_len=512, sub_ratio=1.0, seed=0, test_only=False):

        self.data_name = data_name
        self.total_class = total_class
        self.root_dir = os.path.join(DATA_PATH, data_name)

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sub_ratio = sub_ratio
        self.seed = seed
        self.test_only = test_only

        self.n_classes = int(self.total_class * self.sub_ratio)
        self.class_idx = self._get_subclass()

        if not self._check_exists():
            self._preprocess()

        if not self.test_only:
            self.train_dataset = torch.load(self._train_path)
        else:
            self._train_dataset = None

        self.test_dataset = torch.load(self._test_path)

    def _get_subclass(self):
        np.random.seed(self.seed)  # fix random seed
        class_idx = np.random.permutation(self.total_class)[:self.n_classes]
        return np.sort(class_idx).tolist()

    @property
    def _base_path(self):
        if self.sub_ratio < 1.0:
            base_path = '{}_{}_sub_{:.2f}_seed_{:d}'.format(
                self.data_name, self.tokenizer.name, self.sub_ratio, self.seed)
        else:
            base_path = '{}_{}'.format(self.data_name, self.tokenizer.name)

        return os.path.join(self.root_dir, base_path)

    @property
    def _train_path(self):
        return self._base_path + '_train.pth'

    @property
    def _test_path(self):
        return self._base_path + '_test.pth'

    def _check_exists(self):
        if not self.test_only and not os.path.exists(self._train_path):
            return False
        elif not os.path.exists(self._test_path):
            return False
        else:
            return True

    @abstractmethod
    def _preprocess(self):
        pass


class NewsDataset(BaseDataset):
    def __init__(self, tokenizer, max_len=512, sub_ratio=1.0, seed=0):
        super(NewsDataset, self).__init__('news', 20, tokenizer, max_len, sub_ratio, seed)

    def _preprocess(self):
        print('Pre-processing news dataset...')
        self._preprocess_sub('train')
        self._preprocess_sub('test')

    def _preprocess_sub(self, mode='train'):
        assert mode in ['train', 'test']

        source_path = os.path.join(self.root_dir, '{}.csv'.format(mode))
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        tokens = []
        labels = []

        for line in lines:
            toks = line.split(',')

            if not int(toks[1]) in self.class_idx:  # only selected classes
                continue

            path = os.path.join(self.root_dir, '{}'.format(toks[0]))
            with open(path, encoding='utf-8', errors='ignore') as f:
                text = f.read()

            token = tokenize(self.tokenizer, text, max_len=self.max_len)

            label = self.class_idx.index(int(toks[1]))  # convert to subclass index
            label = torch.tensor(label).long()

            tokens.append(token)
            labels.append(label)

        dataset = create_tensor_dataset(tokens, labels)

        if mode == 'train':
            torch.save(dataset, self._train_path)
        else:
            torch.save(dataset, self._test_path)


class ReviewDataset(BaseDataset):
    def __init__(self, tokenizer, max_len=512, sub_ratio=1.0, seed=0):
        self.split_ratio = 0.7  # split ratio for train/test dataset
        super(ReviewDataset, self).__init__('review', 50, tokenizer, max_len, sub_ratio, seed)

    def _preprocess(self):
        print('Pre-processing review dataset...')
        source_path = os.path.join(self.root_dir, '50EleReviews.json')
        with open(source_path, encoding='utf-8') as f:
            docs = json.load(f)

        np.random.seed(self.seed)  # fix random seed

        train_inds = []
        test_inds = []

        per_class = 1000  # samples are ordered by class
        for cls in self.class_idx:  # only selected classes
            shuffled = np.random.permutation(per_class)
            num = int(self.split_ratio * per_class)

            train_inds += (cls * per_class + shuffled[:num]).tolist()
            test_inds += (cls * per_class + shuffled[num:]).tolist()

        self._preprocess_sub(docs, train_inds, 'train')
        self._preprocess_sub(docs, test_inds, 'test')

    def _preprocess_sub(self, docs, indices, mode='train'):
        assert mode in ['train', 'test']

        tokens = []
        labels = []

        for i in indices:
            token = tokenize(self.tokenizer, docs['X'][i], max_len=self.max_len)

            label = self.class_idx.index(int(docs['y'][i]))  # convert to subclass index
            label = torch.tensor(label).long()

            tokens.append(token)
            labels.append(label)

        dataset = create_tensor_dataset(tokens, labels)

        if mode == 'train':
            torch.save(dataset, self._train_path)
        else:
            torch.save(dataset, self._test_path)


class IMDBDataset(BaseDataset):
    def __init__(self, tokenizer, max_len=512):
        self.class_dict = {'pos': 1, 'neg': 0}
        super(IMDBDataset, self).__init__('imdb', 2, tokenizer, max_len)

    def _preprocess(self):
        print('Pre-processing imdb dataset...')
        source_path = os.path.join(self.root_dir, 'imdb.txt')
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        train_tokens = []
        train_labels = []
        test_tokens = []
        test_labels = []

        for line in lines:
            toks = line.split('\t')

            if len(toks) > 5:  # text contains tab
                text = '\t'.join(toks[2:-2])
                toks = toks[:2] + [text] + toks[-2:]

            token = tokenize(self.tokenizer, toks[2], max_len=self.max_len)

            if toks[3] == 'unsup':
                continue
            else:
                label = self.class_dict[toks[3]]  # convert to class index
                label = torch.tensor(label).long()

            if toks[1] == 'train':
                train_tokens.append(token)
                train_labels.append(label)
            else:
                test_tokens.append(token)
                test_labels.append(label)

        train_dataset = create_tensor_dataset(train_tokens, train_labels)
        test_dataset = create_tensor_dataset(test_tokens, test_labels)

        torch.save(train_dataset, self._train_path)
        torch.save(test_dataset, self._test_path)


class SST2Dataset(BaseDataset):
    def __init__(self, tokenizer, max_len=512):
        super(SST2Dataset, self).__init__('sst2', 2, tokenizer, max_len)

    def _preprocess(self):
        print('Pre-processing sst2 dataset...')
        self._preprocess_sub('train')
        self._preprocess_sub('dev')

    def _preprocess_sub(self, mode='train'):
        assert mode in ['train', 'dev']

        source_path = os.path.join(self.root_dir, 'sst2_{}.tsv'.format(mode))
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        tokens = []
        labels = []

        for line in lines:
            toks = line.split('\t')

            token = tokenize(self.tokenizer, toks[0], max_len=self.max_len)
            label = torch.tensor(int(toks[1])).long()

            tokens.append(token)
            labels.append(label)

        dataset = create_tensor_dataset(tokens, labels)

        if mode == 'train':
            torch.save(dataset, self._train_path)
        else:
            torch.save(dataset, self._test_path)


class FoodDataset(BaseDataset):
    def __init__(self, tokenizer, max_len=512):
        super(FoodDataset, self).__init__('food', 2, tokenizer, max_len)

    def _preprocess(self):
        print('Pre-processing food dataset...')
        self._preprocess_sub('train')
        self._preprocess_sub('test')

    def _preprocess_sub(self, mode='train'):
        assert mode in ['train', 'test']

        source_path = os.path.join(self.root_dir, 'foods_{}.txt'.format(mode))
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        tokens = []
        labels = []

        for line in lines:
            toks = line.split(':')

            if int(toks[1]) == 1:  # pre-defined class 0
                label = 0
            elif int(toks[1]) == 5:  # pre-defined class 1
                label = 1
            else:
                continue

            token = tokenize(self.tokenizer, toks[0], max_len=self.max_len)
            label = torch.tensor(label).long()

            tokens.append(token)
            labels.append(label)

        dataset = create_tensor_dataset(tokens, labels)

        if mode == 'train':
            torch.save(dataset, self._train_path)
        else:
            torch.save(dataset, self._test_path)


class ReutersDataset(BaseDataset):
    def __init__(self, tokenizer, max_len=512):
        super(ReutersDataset, self).__init__('reuters', 2, tokenizer, max_len, test_only=True)

    def _preprocess(self):
        print('Pre-processing reuters dataset...')

        tokens = []
        labels = []

        base_path = os.path.join(self.root_dir, 'reuters_test')
        for fname in os.listdir(base_path):
            path = os.path.join(base_path, fname)
            with open(path, encoding='utf-8', errors='ignore') as f:
                text = f.read()

            token = tokenize(self.tokenizer, text, max_len=self.max_len)
            label = torch.tensor(-1).long()  # OOD class: -1

            tokens.append(token)
            labels.append(label)

        dataset = create_tensor_dataset(tokens, labels)

        torch.save(dataset, self._test_path)

