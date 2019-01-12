
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tools import tensor, struct, show_shapes



def one_hot(label, num_classes):
    t = label.new(label.size(0), num_classes + 1).zero_()
    t.scatter_(1, label.unsqueeze(1), 1)

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
    return errs



# def mask_valid(target, prediction):

#     size_of = lambda t: (t.size(0), t.size(1))
#     sizes = list(map(size_of, [target.location, target.classification, prediction.location, prediction.classification]))
#     assert all_eq (sizes), "total_loss: number of target and prediction differ, " + str(sizes)

#     num_classes = prediction.classification.size(2)

#     pos_mask = (target.classification > 0).unsqueeze(2).expand_as(prediction.location)
#     valid_mask = target.classification >= 0
#     prediction_mask = valid_mask.unsqueeze(2).expand_as(prediction.classification)

#     target = struct(
#         location = target.location[pos_mask], 
#         classification   = target.classification[valid_mask])

#     prediction = struct(
#         location = prediction.location[pos_mask],
#         classification   = prediction.classification[prediction_mask].view(-1, num_classes))

#     return (target, prediction)

# def focal_loss(target, prediction, balance=10, gamma=2, alpha=0.25, eps=1e-6, averaging = False):

#     batch = target.location.size(0)
#     n = prediction.location.size(0) + 1    

#     target, prediction = mask_valid(target, prediction)
    
#     class_loss = focal_loss_bce(target.classification, prediction.classification, gamma=gamma, alpha=alpha).sum()
#     loc_loss = F.smooth_l1_loss(prediction.location, target.location, reduction='sum')

#     return struct(classification = class_loss / (batch * balance), location = loc_loss / batch)


def focal_loss(target, prediction, balance=10, gamma=2, alpha=0.25, eps=1e-6, averaging = False):
    batch = target.location.size(0)
    num_classes = prediction.classification.size(2)

    neg_mask = (target.classification == 0).unsqueeze(2).expand_as(prediction.location)
    invalid_mask = (target.classification < 0).unsqueeze(2).expand_as(prediction.classification)
    
    class_loss = focal_loss_bce(target.classification.clamp(min = 0).view(-1), prediction.classification.view(-1, num_classes), gamma=gamma, alpha=alpha)
    loc_loss = F.smooth_l1_loss(prediction.location.view(-1), target.location.view(-1), reduction='none')

    class_loss = class_loss.view_as(prediction.classification).masked_fill_(invalid_mask, 0)
    loc_loss = loc_loss.view_as(prediction.location).masked_fill_(neg_mask, 0)

    class_loss = class_loss.view(batch, -1).sum(1) / balance
    loc_loss = loc_loss.view(batch, -1).sum(1) 

    parts = struct(classification = class_loss.mean(), location = loc_loss.mean())
    batch = class_loss + loc_loss

    return struct(total = batch.sum(), parts = parts, batch = batch)
