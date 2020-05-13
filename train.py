import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_base_dataset, get_masked_dataset
from models import load_backbone, BaseNet, MaskerNet
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
    parser.add_argument("--keyword_per_class", help='number of keywords for each class',
                        default=10, type=int)
    parser.add_argument("--pretrained_backbone", help='backbone for pre-trained model (None = args.backbone)',
                        default=None, type=str)
    parser.add_argument("--pretrained_path", help='path for the pre-trained model',
                        default=None, type=str)
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

    print('Loading pre-trained backbone networks...')
    backbone, tokenizer = load_backbone(args.backbone)

    print('Initializing dataset and model..')
    if args.model_type == 'base':
        dataset = get_base_dataset(args.dataset, tokenizer,
                                   args.max_len, args.sub_ratio, args.seed)
        model = BaseNet(backbone, dataset.n_classes).to(device)
    else:
        dataset = get_masked_dataset(args, args.dataset, tokenizer, args.keyword_type, args.keyword_per_class,
                                     args.max_len, args.sub_ratio, args.seed)
        model = MaskerNet(backbone, dataset.n_classes, dataset.keyword_num).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    print('Training model..')
    if args.model_type == 'base':
        train_base(args, dataset, model, optimizer)
    else:
        train_masker(args, dataset, model, optimizer)


if __name__ == "__main__":
    main()

