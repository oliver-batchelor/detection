
import torch
import torchvision.ops as torchvision
from tools import table, struct



nms_defaults = struct(
    nms         = 0.5,
    threshold   = 0.05,
    detections  = 500
)

def nms(prediction, params):
    inds = (prediction.confidence >= params.threshold).nonzero().squeeze(1)
    prediction = prediction._index_select(inds)._extend(index = inds)

    inds = torchvision.nms(prediction.bbox, prediction.confidence, params.nms)
    return prediction._index_select(inds)._take(params.detections)


empty_detections = table (
        bbox = torch.FloatTensor(0, 4),
        label = torch.LongTensor(0),
        confidence = torch.FloatTensor(0))