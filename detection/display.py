
import torch
from tools.image import transforms, cv
from tools.image.index_map import default_map

from tools import tensor

def to_rgb(hex):
    return ((hex >> 16) & 255, (hex >> 8)  & 255, (hex >> 0) & 255)

def overlay(image, boxes, labels, confidence=None, classes=None):
    image = image.clone()

    for i in range(0, boxes.size(0)):

        label_class = classes[labels[i]]['name']

        color = to_rgb(label_class['colour'])
        name = label_class['name']

        cv.rectangle(image, boxes[i, :2], boxes[i, 2:], color=color, thickness=2)
        cv.putText(image, name, (boxes[i, 0], boxes[i, 1] + 12), scale = 0.5, color=color, thickness=1)

        if not (confidence is None):
            str = "{:.2f}".format(confidence[i])
            cv.putText(image, str, (boxes[i, 0], boxes[i, 3] - 2), scale = 0.5, color=color, thickness=1)

    return image

def display_batch(batch, cols=6, classes=None):

    images = []
    for b in batch:
        images.append(overlay(b['image'], b['boxes'], b['labels'], classes=classes))

    return tensor.tile_batch(torch.stack(images, 0), cols)
