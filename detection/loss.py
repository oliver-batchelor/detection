
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tools import tensor, Struct



def one_hot(labels, num_classes):
    t = labels.new(labels.size(0), num_classes + 1).zero_()
    t.scatter_(1, labels.unsqueeze(1), 1)

    return t[:, 1:]

def all_eq(xs):
    return all(map(lambda x: x == xs[0], xs))




def focal_loss_softmax(class_target, class_pred, gamma=2, alpha=0.25, eps = 1e-6):
    #ce = F.cross_entropy(class_pred, class_target, size_average = False)

    p = F.softmax(class_pred, 1).clamp(eps, 1 - eps)
    p = p.gather(1, class_target.unsqueeze(1))

    errs = -(1 - p).pow(gamma) * p.log()

    return errs.sum()


def focal_loss_bce(class_target, class_pred, gamma=2, alpha=0.25, eps=1e-6):

    num_classes = class_pred.size(1)
    y = one_hot(class_target.detach(), num_classes).float()
    y_inv = 1 - y

    p_t = y * class_pred + y_inv * (1 - class_pred)
    a_t = y * alpha      + y_inv * (1 - alpha)

    p_t = p_t.clamp(min=eps, max=1-eps)

    errs = -a_t * (1 - p_t).pow(gamma) * p_t.log()
    return errs.sum()



def mask_valid(target, prediction):
    # loc_target, class_target =  target
    # loc_pred, class_pred = prediction

    size_of = lambda t: (t.size(0), t.size(1))
    sizes = list(map(size_of, [target.locations, target.classes, prediction.locations, prediction.classes]))
    assert all_eq (sizes), "total_loss: number of target and prediction differ, " + str(sizes)

    num_classes = prediction.classes.size(2)

    pos_mask = (target.classes > 0).unsqueeze(2).expand_as(prediction.locations)
    valid_mask = target.classes >= 0
    prediction_mask = valid_mask.unsqueeze(2).expand_as(prediction.classes)

    target = Struct(
        locations = target.locations[pos_mask], 
        classes   = target.classes[valid_mask])

    prediction = Struct(
        locations = prediction.locations[pos_mask],
        classes   = prediction.classes[prediction_mask].view(-1, num_classes))

    return (target, prediction)


def total_bce(target, prediction, balance=5, gamma=2, alpha=0.25, eps=1e-6):
    target, prediction = mask_valid(target, prediction)

    class_loss = focal_loss_bce(target.classes, prediction.classes, gamma=gamma, alpha=alpha)
    loc_loss = F.smooth_l1_loss(prediction.locations, target.locations, reduction='sum')

    return Struct(classes = class_loss / balance, locations = loc_loss)
