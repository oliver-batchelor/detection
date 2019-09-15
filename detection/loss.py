
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tools import tensor, struct, show_shapes


# makes a one_hot vector from class labels
def one_hot(label, num_classes):
    t = label.new(label.size(0), num_classes).zero_()
    return t.scatter_(1, label.unsqueeze(1), 1)

# makes a one_hot vector from class labels with an 'ignored' case as 0 (which is trimmed)
def one_hot_with_ignored(label, num_classes):
    return one_hot(label, num_classes + 1)[:, 1:]



def all_eq(xs):
    return all(map(lambda x: x == xs[0], xs))



def focal_loss_label(target_labels, pred, class_weights, gamma=2, eps=1e-6):
    num_classes = pred.size(1)
    target = one_hot_with_ignored(target_labels.detach(), num_classes).float()

    alpha = class_weights[target_labels].unsqueeze(1)
    return focal_loss_bce(target, pred, alpha, gamma=gamma, eps=eps)


def focal_loss_bce(target, pred, alpha, gamma=2, eps=1e-6):
    target_inv = 1 - target

    p_t = target * pred + target_inv * (1 - pred)
    a_t = target * alpha      + target_inv * (1 - alpha)

    p_t = p_t.clamp(min=eps, max=1-eps)

    errs = -a_t * (1 - p_t).pow(gamma) * p_t.log()
    return errs



def batch_focal_loss(target, prediction, class_weights, balance=4, gamma=2, eps=1e-6):
    batch = target.location.size(0)
    num_classes = prediction.classification.size(2)

    class_weights = prediction.classification.new([0.0, *class_weights])

    neg_mask = (target.classification == 0).unsqueeze(2).expand_as(prediction.location)
    invalid_mask = (target.classification < 0).unsqueeze(2).expand_as(prediction.classification)
    
    class_loss = focal_loss_label(target.classification.clamp(min = 0).view(-1), 
        prediction.classification.view(-1, num_classes), class_weights=class_weights, gamma=gamma)

    loc_loss = F.smooth_l1_loss(prediction.location.view(-1), target.location.view(-1), reduction='none')

    class_loss = class_loss.view_as(prediction.classification).masked_fill_(invalid_mask, 0)
    loc_loss = loc_loss.view_as(prediction.location).masked_fill_(neg_mask, 0)

    class_loss = class_loss.view(batch, -1).sum(1) / balance
    loc_loss = loc_loss.view(batch, -1).sum(1) 

    parts = struct(classification = class_loss.sum(), location = loc_loss.sum())
    batch = class_loss + loc_loss

    return struct(total = batch.sum(), parts = parts, batch = batch)


