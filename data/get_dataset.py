import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.base_dataset import NewsDataset, ReviewDataset, IMDBDataset, SST2Dataset, FoodDataset, ReutersDataset
from data.masked_dataset import MaskedDataset
from models import load_backbone

from common import SAVE_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_base_dataset(data_name, tokenizer,
                     max_len=512, sub_ratio=1.0, seed=0):

    if data_name == 'news':
        dataset = NewsDataset(tokenizer, max_len, sub_ratio, seed)
    elif data_name == 'review':
        dataset = ReviewDataset(tokenizer, max_len, sub_ratio, seed)
    elif data_name == 'imdb':
        dataset = IMDBDataset(tokenizer, max_len)
    elif data_name == 'sst2':
        dataset = SST2Dataset(tokenizer, max_len)
    elif data_name == 'food':
        dataset = FoodDataset(tokenizer, max_len)
    elif data_name == 'reuters':
        dataset = ReutersDataset(tokenizer, max_len)
    else:
        raise ValueError('No matching dataset')

    return dataset


def get_masked_dataset(args, data_name, tokenizer, keyword_type, keyword_per_class,
                       max_len=512, sub_ratio=1.0, seed=0):

    dataset = get_base_dataset(data_name, tokenizer, max_len, sub_ratio, seed)  # base dataset

    keyword_path = '{}_{}.pth'.format(keyword_type, keyword_per_class)
    keyword_path = os.path.join(dataset.root_dir, keyword_path)

    if os.path.exists(keyword_path):
        keyword = torch.load(keyword_path)
    else:
        keyword = get_keyword(args, dataset, tokenizer, keyword_type, keyword_per_class)
        torch.save(keyword, keyword_path)

    masked_dataset = MaskedDataset(dataset, keyword)

    return masked_dataset


class Keyword(object):
    def __init__(self, keyword_type, keyword):
        self.keyword_type = keyword_type
        self.keyword = keyword

    def __len__(self):
        return len(self.keyword)


def get_keyword(args, dataset, tokenizer, keyword_type, keyword_per_class):
    if keyword_type == 'tfidf':
        keyword = get_tfidf_keyword(dataset, keyword_per_class)
        keyword = Keyword('tfidf', keyword)

    elif keyword_type == 'attention':
        if args.pretrained_backbone is None:
            args.pretrained_backbone = args.backbone

        attn_model, _ = load_backbone(args.pretrained_backbone, output_attentions=True)
        attn_model.to(device)  # only backbone

        if args.pretrained_path is not None:
            ckpt = torch.load(os.path.join(SAVE_PATH, args.pretrained_path))
            attn_model.load_state_dict(ckpt, strict=False)  # assume ckpt is state_dict
        else:
            print('Warning! Pre-trained model is not specified. Use random network.')

        if torch.cuda.device_count() > 1:
            attn_model = nn.DataParallel(attn_model)

        keyword = get_attention_keyword(dataset, attn_model, keyword_per_class)
        keyword = Keyword('attention', keyword)

        del attn_model  # free GPU memory

    else:  # random
        keyword = list(tokenizer.vocab.values())  # all words
        keyword = Keyword('random', keyword)

    return keyword


def get_tfidf_keyword(dataset, keyword_per_class=10):
    raise NotImplementedError


def get_attention_keyword(dataset, attn_model, keyword_per_class=10):
    loader = DataLoader(dataset.train_dataset, shuffle=False,
                        batch_size=32, num_workers=4)

    SPECIAL_TOKENS = dataset.tokenizer.all_special_ids

    vocab_size = len(dataset.tokenizer)

    attn_score = torch.zeros(vocab_size)
    attn_freq = torch.zeros(vocab_size)

    for _, (tokens, _) in enumerate(loader):
        tokens = tokens.to(device)

        with torch.no_grad():
            out_h, out_p, attention_layers = attn_model(tokens)

        attention = attention_layers[-1]  # attention of final layer (batch_size, num_heads, max_len, max_len)
        attention = attention.sum(dim=1)  # sum over attention heads (batch_size, max_len, max_len)

        for i in range(attention.size(0)):  # batch_size
            for j in range(attention.size(-1)):  # max_len
                token = tokens[i][j].item()

                if token in SPECIAL_TOKENS:  # skip special token
                    continue

                score = attention[i][0][j]  # 1st token = CLS token

                attn_score[token] += score.item()
                attn_freq[token] += 1

    for tok in range(vocab_size):
        attn_score[tok] /= attn_freq[tok]  # normalize by frequency

    num = keyword_per_class * dataset.n_classes  # number of total keywords
    keyword = attn_score.argsort(descending=True)[:num].tolist()

    return keyword

