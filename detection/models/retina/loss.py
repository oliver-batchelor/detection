
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tools import tensor, struct, shape
from detection import box

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


def l1(target, prediction, class_target):

    loss = F.smooth_l1_loss(prediction.view(-1, 4), target.view(-1, 4), reduction='none')
       
    neg_mask = (class_target == 0).unsqueeze(2).expand_as(prediction)
    return loss.masked_fill_(neg_mask.view_as(loss), 0).sum()  

def giou(target, prediction, class_target):

    giou = box.giou(prediction.view(-1, 4), target.view(-1, 4))
    neg_mask = (class_target == 0).view_as(giou)

    # Constant 0.05 is to make the magnitude roughly equivalent with l1 loss
    return 0.05 * (1 - giou).masked_fill_(neg_mask, 0).sum() 

def iou(target, prediction, class_target):

    iou = box.iou(prediction.view(-1, 4), target.view(-1, 4))
    neg_mask = (class_target == 0).view_as(iou)

    return 0.05 * (1 - iou).masked_fill_(neg_mask, 0).sum() 


def class_loss(target, prediction, class_weights,  gamma=2, eps=1e-6):
    batch, _, num_classes = prediction.shape

    class_weights = prediction.new([0.0, *class_weights])
    invalid_mask = (target < 0).unsqueeze(2).expand_as(prediction)
    
    loss = focal_loss_label(target.clamp(min = 0).view(-1), 
        prediction.view(-1, num_classes), class_weights=class_weights, gamma=gamma)\

    return loss.masked_fill_(invalid_mask.view_as(loss), 0).sum()


