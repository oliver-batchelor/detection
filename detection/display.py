
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


    def overlay_prediction():
        for prediction in eval.prediction._sequence():
            if prediction.confidence < threshold:
                break

            label_class = classes[prediction.label]['name']
            draw_box(image, prediction.bbox, scale=scale, confidence=prediction.confidence, name=label_class['name'], color=to_rgb(label_class['colour']))


    def overlay_target():

        for target in eval.target._sequence():
            label_class = classes[target.label]['name']
            draw_box(image, target.bbox, scale=scale, name=label_class['name'], color=to_rgb(label_class['colour']))


    def overlay_anchors():
        overlay_target()

        for anchor in eval.anchors:
            label_class = classes[anchor.label]['name']
            draw_box(image, anchor.bbox, scale=scale, color=to_rgb(label_class['colour']), thickness=1)


    def overlay_matches():
        unmatched = dict(enumerate(eval.target._sequence()))

        # print(unmatched)

        for m in eval.matches:
            if m.confidence < threshold: break

            if m.match is not None:
                k, _ = m.match
                del unmatched[k]


        for (i, target) in enumerate(eval.target._sequence()):
            label_class = classes[target.label]['name']
            color = (255, 0, 0) if i in unmatched else (0, 255, 0)

            draw_box(image, target.bbox, scale=scale, name=label_class['name'], color=color)

        for m in eval.matches:
            if m.confidence < threshold: break

            color = (255, 0, 0)
            if m.match is not None:
                color = (0, 255, 0)

            print(m.bbox)

            label_class = classes[m.label]['name']
            draw_box(image, m.bbox, scale=scale, color=color, confidence=m.confidence, name=label_class['name'], thickness=1)



    target = {
        'matches'       : overlay_matches,
        'prediction'   : overlay_prediction,
        'anchors'       : overlay_anchors,
        'target'       : overlay_target
    }

    assert (mode in target), "overlay: invalid mode " + mode + ", expected one of " + str(target.keys())
    target[mode]()

    return image



def overlay_batch(batch, mode='target', scale=1.0, threshold = 0.5, cols=6, classes=None):

    images = []
    for eval in batch:
        images.append(overlay(eval, scale=scale, mode=mode, threshold=threshold, classes=classes))

    return tensor.tile_batch(torch.stack(images, 0), cols)
