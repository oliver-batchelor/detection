
import torch
from tools.image import transforms, cv
from tools.image.index_map import default_map

from tools import tensor

def to_rgb(hex):
    return ((hex >> 16) & 255, (hex >> 8)  & 255, (hex >> 0) & 255)


def draw_box(image, box, scale=1.0, name=None, confidence=None, thickness=2, color=(255, 0, 0), text_color=None):

    text_color = text_color or color
    cv.rectangle(image, box[:2], box[2:], color=color, thickness=int(thickness * scale))

    if not (name is None):
        cv.putText(image, name, (box[0], box[1] + int(8 * scale)), scale = 0.7 * scale, color=text_color, thickness=int(1*scale))

    if not (confidence is None):
        str = "{:.2f}".format(confidence)
        cv.putText(image, str, (box[0], box[3] - 2), scale = 0.7 * scale, color=text_color, thickness=int(1*scale))


def overlay(eval, mode='target', threshold = 0.5, scale=1.0, classes=None):
    image = eval.image.clone()

    cv.putText(image, eval.file, (0, int(12 * scale)), scale = scale, color=(64, 64, 192), thickness=int(1*scale))
    cv.putText(image, "mAP@0.5 " + str(eval.mAP), (0, int(24 * scale)), scale = scale, color=(64, 64, 192), thickness=int(1*scale))


    def overlay_predictions():
        for (label, box, confidence) in eval.predictions:
            if confidence < threshold:
                break

            label_class = classes[label]['name']
            draw_box(image, box, scale=scale, confidence=confidence, name=label_class['name'], color=to_rgb(label_class['colour']))


    def overlay_targets():
        for (label, box) in eval.targets:
            label_class = classes[label]['name']
            draw_box(image, box, scale=scale, name=label_class['name'], color=to_rgb(label_class['colour']))


    def overlay_anchors():
        overlay_targets()

        for (label, box) in eval.anchors:
            label_class = classes[label]['name']
            draw_box(image, box, scale=scale, color=to_rgb(label_class['colour']), thickness=1)


    def overlay_matches():
        unmatched = dict(enumerate(eval.targets))

        for m in eval.matches:
            if m.confidence < threshold: break

            if m.match is not None:
                del unmatched[m.match]


        for (i, (label, box)) in enumerate(eval.targets):
            label_class = classes[label]['name']
            color = (255, 0, 0) if i in unmatched else (0, 255, 0)

            draw_box(image, box, scale=scale, name=label_class['name'], color=color)

        for m in eval.matches:
            if m.confidence < threshold: break

            color = (255, 0, 0)
            if m.match is not None:
                color = (0, 255, 0)

            label_class = classes[m.label]['name']
            draw_box(image, m.box, scale=scale, color=color, confidence=m.confidence, name=label_class['name'], thickness=1)



    targets = {
        'matches'       : overlay_matches,
        'predictions'   : overlay_predictions,
        'anchors'       : overlay_anchors,
        'targets'       : overlay_targets
    }

    assert (mode in targets), "overlay: invalid mode " + mode + ", expected one of " + str(targets.keys())
    targets[mode]()

    return image



def overlay_batch(batch, mode='target', scale=1.0, threshold = 0.5, cols=6, classes=None):

    images = []
    for eval in batch:
        images.append(overlay(eval, scale=scale, mode=mode, threshold=threshold, classes=classes))

    return tensor.tile_batch(torch.stack(images, 0), cols)
