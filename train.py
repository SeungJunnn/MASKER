import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import get_base_dataset, get_masked_dataset
from models import load_backbone, BaseNet, MaskerNet
from training import train_base, train_masker
from evals import test_acc

from common import CKPT_PATH, parse_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args(mode='train')

    if args.train_type == 'masker':
        if args.keyword_type == 'random':
            args.batch_size = 4
        else:
            args.batch_size = 16
    else:
        args.batch_size = 32

    print('Loading pre-trained backbone network...')
    backbone, tokenizer = load_backbone(args.backbone)

    print('Initializing dataset and model...')
    if args.train_type == 'base':
        dataset = get_base_dataset(args.dataset, tokenizer, args.split_ratio, args.seed)
        model = BaseNet(args.backbone, backbone, dataset.n_classes).to(device)
    else:
        dataset = get_masked_dataset(args, args.dataset, tokenizer, args.keyword_type, args.keyword_per_class,
                                     args.split_ratio, args.seed)
        model = MaskerNet(args.backbone, backbone, dataset.n_classes, dataset.keyword_num).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    #optimizer = optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr' : 5e-5} ], lr=1e-3, eps=1e-8)

    train_loader = DataLoader(dataset.train_dataset, shuffle=True, drop_last=True,
                              batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(dataset.test_dataset, shuffle=False,
                             batch_size=args.batch_size, num_workers=4)

    print('Training model...')
    for epoch in range(1, args.epochs + 1):
        if args.train_type == 'base':
            train_base(args, train_loader, model, optimizer, epoch)
        else:
            train_masker(args, train_loader, model, optimizer, epoch)

        acc = test_acc(test_loader, model)
        print('test acc: {:.2f}'.format(acc))

    if isinstance(model, nn.DataParallel):
        model = model.module

    os.makedirs(os.path.join(CKPT_PATH, dataset.data_name), exist_ok=True)
    save_path = os.path.join(CKPT_PATH, dataset.data_name, dataset.base_path + '_model.pth')
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()

