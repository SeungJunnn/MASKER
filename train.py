import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_base_dataset, get_masked_dataset
from models import BaseNet, MaskerNet
from training import train_base, train_masker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset arguments
    parser.add_argument("--dataset", help='dataset (news|review|imdb|etc.)',
                        required=True, type=str)
    parser.add_argument("--max_len", help='maximum length of sentences',
                        default=512, type=int)
    parser.add_argument("--sub_ratio", help='subsample ratio for ID/OOD sets',
                        default=1.0, type=float)
    parser.add_argument("--seed", help='random seed (used in dataset subsample)',
                        default=0, type=int)
    # model arguments
    parser.add_argument("--model_type", help='model type (base|masker)',
                        choices=['base', 'masker'],
                        default='mask_ent', type=str)
    parser.add_argument("--backbone", help='backbone network',
                        choices=['bert', 'roberta'],
                        default='bert', type=str)
    parser.add_argument("--classifier_type", help='classifier type (softmax|sigmoid)',
                        choices=['softmax', 'sigmoid'],
                        default='sigmoid', type=str)
    # training arguments
    parser.add_argument("--epochs", help='training epochs',
                        default=10, type=int)
    parser.add_argument("--batch_size", help='batch size',
                        default=16, type=int)
    parser.add_argument("--keyword_type", help='keyword type (random|tfidf|attention|etc.)',
                        choices=['random', 'tfidf', 'attention'],
                        default='attention', type=str)
    parser.add_argument("--keyword_num", help='number of keywords for each sentence',
                        default=10, type=int)
    parser.add_argument("--lambda_ssl", help='weight for keyword reconstruction loss',
                        default=0.001, type=float)
    parser.add_argument("--lambda_ood", help='weight for outlier regularization loss',
                        default=0.0001, type=float)

    return parser.parse_args()


def main():
    args = parse_args()

    print('dataset: {}'.format(args.dataset))
    print('sub_ratio: {}'.format(args.sub_ratio))
    print('model_type: {}'.format(args.model_type))
    print('backbone: {}'.format(args.backbone))
    print('classifier_type: {}'.format(args.classifier_type))

    if args.model_type == 'masker':
        if args.keyword_type == 'random':
            args.batch_size = 4
        else:
            args.batch_size = 16
    else:
        args.batch_size = 32

    ### load backbone and tokenizer ###
    if args.backbone == 'bert':
        from transformers import BertModel, BertTokenizer
        backbone = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.name = 'bert-base-uncased'
    elif args.backbone == 'roberta':
        from transformers import RobertaModel, RobertaTokenizer
        backbone = RobertaModel.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer.name = 'roberta-base'
    else:
        raise ValueError('No matching backbone network')

    ### create model and dataset ###
    if args.model_type == 'base':
        dataset = get_base_dataset(args.dataset, tokenizer,
                                   args.max_len, args.sub_ratio, args.seed)
        model = BaseNet(backbone, dataset.n_classes).to(device)
    else:
        dataset = get_masked_dataset(args.dataset, tokenizer, args.keyword_type, args.keyword_num,
                                     args.max_len, args.sub_ratio, args.seed)
        model = MaskerNet(backbone, dataset.n_classes, dataset.keyword_num).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    ### train model ###
    if args.model_type == 'base':
        train_base(args, dataset, model, optimizer)
    else:
        train_masker(args, dataset, model, optimizer)


if __name__ == "__main__":
    main()

