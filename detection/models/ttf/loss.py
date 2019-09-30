import torch
from detection import box


def class_loss(target, prediction, class_weights):
    """
    Focal loss variant of BCE as used in CornerNet and CenterNet.
    """

    print(target.shape, prediction.shape)

    # As per RetinaNet focal loss - if heatmap == 1
    pos_loss = -prediction.log() * (1 - prediction).pow(2)

    # Negative penalty very small around gaussian near centre
    neg_weights = (1 - target).pow(4) 
    neg_loss = -(1 - prediction).log() * prediction.pow(2) * neg_weights

    return torch.where(target == 1, pos_loss, neg_loss)

def giou(target, prediction, weight):
    assert target.shape == prediction.shape, str(target.shape) + " vs. " + str(prediction.shape)

    giou = box.giou(prediction.view(-1, 4), target.view(-1, 4))
    return (1 - giou).mul_(weight.view(-1)).sum()