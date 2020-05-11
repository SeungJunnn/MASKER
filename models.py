import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseNet(nn.Module):
    """ Base network """

    def __init__(self, backbone, n_classes):
        super(BaseNet, self).__init__()
        self.backbone = backbone
        self.net_cls = nn.Linear(768, n_classes)  # classification layer

    def forward(self, x):
        attention_mask = (x > 0).float()
        out_h, out_p = self.backbone(x, attention_mask)  # hidden, pooled

        out_cls = self.net_cls(out_p)
        return out_cls


class MaskerNet(nn.Module):
    """ Makser network """

    def __init__(self, backbone, n_classes, vocab_size):
        super(MaskerNet, self).__init__()
        self.backbone = backbone
        self.net_cls = nn.Linear(768, n_classes)  # classification layer
        self.net_ssl = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, vocab_size),
        )  # self-supervision layer

    def forward(self, x, training=False):
        if training:  # training mode
            x_orig, x_mask, x_ood = x.chunk(3, dim=1)  # (original, masked, outlier)
            attention_mask = (x_orig > 0).float()

            out_cls = self.backbone(x_orig, attention_mask)[1]  # pooled feature
            out_cls = self.net_cls(out_cls)  # classification

            out_ssl = self.backbone(x_mask, attention_mask)[0]  # hidden feature
            out_ssl = self.net_ssl(out_ssl)  # self-supervision

            out_ood = self.backbone(x_ood, attention_mask)[1]  # pooled feature
            out_ood = self.net_cls(out_ood)  # classification (outlier)

            return out_cls, out_ssl, out_ood

        else:  # inference mode
            attention_mask = (x > 0).float()

            out_cls = self.backbone(x, attention_mask)[1]  # pooled feature
            out_cls = self.net_cls(out_cls)  # classification

            return out_cls

