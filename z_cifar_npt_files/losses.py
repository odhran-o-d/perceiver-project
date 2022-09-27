
import torch
import torch.nn.functional as F


def multi_class_multi_label_CE_loss(pred, target: torch.float):
    return F.cross_entropy(pred, target)


def single_label_weighted_ce_loss(pred, target: torch.long):
    # based on https://openreview.net/attachment?id=jyd4Lyjr2iB&name=supplementary_material
    losses = F.cross_entropy(pred, target, reduction='none')
    proportion = torch.bincount(target)/len(target)
    broadcast = torch.stack([proportion[i] for i in target])
    loss = (torch.sum(losses / broadcast)) / len(target)
    return loss


def negative_log_likelihood(pred, target):
    return F.nll_loss(pred, target)


def mean_squared_error(pred, target, num_classes):
    y_hot = torch.squeeze(F.one_hot(target.long(), num_classes=num_classes))
    return F.mse_loss(pred, y_hot.float())
