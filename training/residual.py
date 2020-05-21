import torch
import torch.nn as nn
import torch.nn.functional as F
from training.common import AverageMeter, one_hot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_residual(args, loader, model, biased_model, optimizer, epoch=0):
    model.train()

    if isinstance(model, nn.DataParallel):
        n_classes = model.module.n_classes
    else:
        n_classes = model.n_classes

    losses = dict()
    losses['cls'] = AverageMeter()

    for i, (tokens, labels) in enumerate(loader):
        batch_size = tokens.size(0)
        tokens = tokens.to(device)
        labels = labels.to(device)

        labels = labels.squeeze(1)  # (B)

        with torch.no_grad():
            out_biased = biased_model(tokens)  # (B, C)
        out_cls = model(tokens)  # (B, C)

        # classification loss
        if args.classifier_type == 'softmax':
            p_mult = F.softmax(out_cls, dim=1) * F.softmax(out_biased, dim=1)  # multiply probs
            loss_cls = F.nll_loss(torch.log(p_mult), labels)  # log probs, not logits
        else:
            p_mult = torch.sigmoid(out_cls) * torch.sigmoid(out_biased)  # multiply probs
            labels = one_hot(labels, n_classes=n_classes)
            loss_cls = F.binary_cross_entropy(p_mult, labels)  # probs, not logits

        # total loss
        loss = loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses['cls'].update(loss_cls.item(), batch_size)

    print('[Epoch %2d] [LossC %f]' %
          (epoch, losses['cls'].average))

