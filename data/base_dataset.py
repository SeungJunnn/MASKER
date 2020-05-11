import os
from abc import *

import torch
from torch.utils.data import TensorDataset
import numpy as np

HOME = os.path.expanduser('~')
DATA_PATH = os.path.join(HOME, 'data_masker')


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


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, total_class, tokenizer, max_len=512, sub_ratio=1.0, seed=0):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sub_ratio = sub_ratio
        self.seed = seed

        self.total_class = total_class
        self.n_classes = int(self.total_class * self.sub_ratio)
        self.class_idx = self._get_subclass()

        if not self._check_exists():
            self._preprocess()

        self.train_dataset = torch.load(self._train_path)
        self.test_dataset = torch.load(self._test_path)

    @property
    @abstractmethod
    def _train_path(self):
        pass

    @property
    @abstractmethod
    def _test_path(self):
        pass

    def _check_exists(self):
        if os.path.exists(self._train_path) and os.path.exists(self._test_path):
            return True
        else:
            return False

    def _get_subclass(self):
        np.random.seed(self.seed)  # fix random seed
        class_idx = np.random.permutation(self.total_class)[:self.n_classes]
        return np.sort(class_idx).tolist()

    @abstractmethod
    def _preprocess(self):
        pass


class NewsDataset(BaseDataset):
    def __init__(self, tokenizer, max_len=512, sub_ratio=1.0, seed=0):
        total_class = 20
        super(NewsDataset, self).__init__(total_class, tokenizer, max_len, sub_ratio, seed)

    @property
    def _train_path(self):
        train_path = 'news/news_{}_sub_{:.2f}_seed_{:d}_train.pth'.format(
            self.tokenizer.name, self.sub_ratio, self.seed)
        return os.path.join(DATA_PATH, train_path)

    @property
    def _test_path(self):
        test_path = 'news/news_{}_sub_{:.2f}_seed_{:d}_test.pth'.format(
            self.tokenizer.name, self.sub_ratio, self.seed)
        return os.path.join(DATA_PATH, test_path)

    def _preprocess(self):
        self._preprocess_sub('train')
        self._preprocess_sub('test')

    def _preprocess_sub(self, mode='train'):
        assert mode in ['train', 'test']

        source_path = os.path.join(DATA_PATH, 'news/{}.csv'.format(mode))
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        tokens = []
        labels = []

        for line in lines:
            toks = line.split(',')

            if not int(toks[1]) in self.class_idx:
                continue

            path = os.path.join(DATA_PATH, 'news/{}'.format(toks[0]))
            with open(path, encoding='utf-8', errors='ignore') as f:
                text = f.read()

            token = tokenize(self.tokenizer, text, max_len=self.max_len)

            label = self.class_idx.index(int(toks[1]))  # convert to subclass index
            label = torch.tensor(label).long()

            tokens.append(token)
            labels.append(label)

        assert len(tokens) == len(labels)

        tokens = torch.stack(tokens)
        labels = torch.stack(labels).unsqueeze(1)

        dataset = TensorDataset(tokens, labels)

        if mode == 'train':
            torch.save(dataset, self._train_path)
        else:
            torch.save(dataset, self._test_path)


class IMDBDataset(BaseDataset):
    def __init__(self, tokenizer, max_len=512):
        total_class = 2
        super(IMDBDataset, self).__init__(total_class, tokenizer, max_len)
        self.class_dict = {'pos': 1, 'neg': 0}

    @property
    def _train_path(self):
        return os.path.join(DATA_PATH, 'imdb/imdb_train.pth')

    @property
    def _test_path(self):
        return os.path.join(DATA_PATH, 'imdb/imdb_test.pth')

    def _preprocess(self):
        source_path = os.path.join(DATA_PATH, 'imdb/imdb.txt')
        with open(source_path, encoding='utf-8') as f:
            lines = f.readlines()

        train_tokens = []
        train_labels = []
        test_tokens = []
        test_labels = []

        for line in lines:
            toks = line.split('\t')

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

        assert len(train_tokens) == len(train_labels)
        assert len(test_tokens) == len(test_labels)

        train_tokens = torch.stack(train_tokens)
        train_labels = torch.stack(train_labels).unsqueeze(1)
        test_tokens = torch.stack(test_tokens)
        test_labels = torch.stack(test_labels).unsqueeze(1)

        train_dataset = TensorDataset(train_tokens, train_labels)
        test_dataset = TensorDataset(test_tokens, test_labels)

        torch.save(train_dataset, self._train_path)
        torch.save(test_dataset, self._test_path)

