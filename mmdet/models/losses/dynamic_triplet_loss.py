import torch
import torch.nn as nn
from ..builder import LOSSES


@LOSSES.register_module
class DynamicTripletLoss(nn.Module):
    def __init__(self,
                 p=2,
                 sigma=2,
                 alpha=0.6,
                 min_margin=0.2,
                 loss_weight=1.0
                 ):
        super().__init__()
        self.p = p
        self.sigma = sigma
        self.alpha = alpha
        self.min_margin = min_margin
        self.loss_weight = loss_weight

    def forward(self, anchors, positives, negatives, weight_norm_ratio):
        anchor_positive_dists = torch.norm((anchors - positives), p=self.p, dim=-1)
        anchor_negative_dists = torch.norm((anchors - negatives), p=self.p, dim=-1)
        dynamic_margin = self.alpha * torch.exp(-1 * (weight_norm_ratio - 1) / self.sigma)
        # min_margin
        min_dists = self.min_margin * torch.ones_like(dynamic_margin)
        # 小于min_margin的时候补min_margin
        dynamic_margin = torch.where(dynamic_margin < self.min_margin, min_dists, dynamic_margin)
        # 计算d(a,p)-d(a,n) + margin
        subs_dists = anchor_positive_dists - anchor_negative_dists + dynamic_margin
        # zeros like
        zero_dists = torch.zeros_like(subs_dists)
        # max(0, d(a,p)-d(a,n) + margin)
        dists = torch.where(subs_dists < 0, zero_dists, subs_dists)
        # print(dists)
        # print('+++++')
        if len(dists) == 0:
            length = 1
        else:
            length = len(dists)

        return self.loss_weight * torch.sum(dists) / length
