import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from training.common import AverageMeter, one_hot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_base(args, dataset, model, optimizer):
    loader = DataLoader(dataset.train_dataset, shuffle=True, drop_last=True,
                        batch_size=args.batch_size, num_workers=4)

    for epoch in range(1, args.epochs + 1):
        losses = dict()
        losses['cls'] = AverageMeter()

        for i, (tokens, labels) in enumerate(loader):
            batch_size = tokens.size(0)
            tokens = tokens.to(device)
            labels = labels.to(device)

            labels = labels.squeeze(1)  # (B)

            out_cls = model(tokens)  # (B, C)

            # classification loss
            if args.classifier_type == 'softmax':
                loss_cls = F.cross_entropy(out_cls, labels)
            else:
                labels = one_hot(labels, n_classes=dataset.n_classes)
                loss_cls = F.binary_cross_entropy_with_logits(out_cls, labels)

            # total loss
            loss = loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses['cls'].update(loss_cls.item(), batch_size)

        print('[Epoch %2d] [LossC %f]' %
              (epoch, losses['cls'].average))

