import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from training.common import AverageMeter, one_hot, uniform_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_masker(args, dataset, model, optimizer):
    loader = DataLoader(dataset.train_dataset, shuffle=True, drop_last=True,
                        batch_size=args.batch_size, num_workers=4)

    for epoch in range(1, args.epochs + 1):
        losses = dict()
        losses['cls'] = AverageMeter()
        losses['ssl'] = AverageMeter()
        losses['ood'] = AverageMeter()

        for i, (tokens, labels) in enumerate(loader):
            batch_size = tokens.size(0)
            tokens = tokens.to(device)
            labels = labels.to(device)

            labels_ssl = labels[:, :-1]  # self-sup labels (B, K)
            labels_cls = labels[:, -1]  # class labels (B)

            out_cls, out_ssl, out_ood = model(tokens, training=True)

            # classification loss
            if args.classifier_type == 'softmax':
                loss_cls = F.cross_entropy(out_cls, labels_cls)
            else:
                labels_cls = one_hot(labels_cls, n_classes=dataset.n_classes)
                loss_cls = F.binary_cross_entropy_with_logits(out_cls, labels_cls)

            # self-supervision loss
            out_ssl = out_ssl.permute(0, 2, 1)
            loss_ssl = F.cross_entropy(out_ssl, labels_ssl, ignore_index=-1)  # ignore non-masks (-1)
            loss_ssl = loss_ssl * args.lambda_ssl

            # outlier regularization loss
            out_ood = F.log_softmax(out_ood, dim=1)  # log-probs
            unif = uniform_labels(labels, n_classes=dataset.n_classes)
            loss_ood = F.kl_div(out_ood, unif)
            loss_ood = loss_ood * args.lambda_ood

            # total loss
            loss = loss_cls + loss_ssl + loss_ood

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses['cls'].update(loss_cls.item(), batch_size)
            losses['ssl'].update(loss_ssl.item(), batch_size)
            losses['ood'].update(loss_ood.item(), batch_size)

        print('[Epoch %2d] [LossCLS %f] [LossSSL %f] [LossOOD %f]' %
              (epoch, losses['cls'].average, losses['ssl'].average, losses['ood'].average))

