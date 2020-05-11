from data.base_dataset import *
from data.masked_dataset import MaskedDataset


def get_base_dataset(data_name, tokenizer,
                     max_len=512, sub_ratio=1.0, seed=0):

    if data_name == 'news':
        dataset = NewsDataset(tokenizer, max_len, sub_ratio, seed)
    elif data_name == 'review':
        raise NotImplementedError
    elif data_name == 'imdb':
        dataset = IMDBDataset(tokenizer, max_len)
    elif data_name == 'amazon':
        raise NotImplementedError
    elif data_name == 'food':
        raise NotImplementedError
    elif data_name == 'sst2':
        raise NotImplementedError
    elif data_name == 'reuters':
        raise NotImplementedError
    else:
        raise ValueError('No matching dataset')

    return dataset


def get_masked_dataset(data_name, tokenizer, keyword_type, keyword_num,
                       max_len=512, sub_ratio=1.0, seed=0):

    dataset = get_base_dataset(data_name, tokenizer, max_len, sub_ratio, seed)
    masked_dataset = MaskedDataset(dataset, keyword_type, keyword_num)

    return masked_dataset



