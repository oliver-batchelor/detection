
import json
from dataset import annotate

import argparse

from tools import struct, to_structs
from detection import evaluate

import dateutil.parser as date


def decode_action(action):

    if action.tag == 'HistUndo':
        return struct(action = 'undo')

    elif action.tag == 'HistRedo':
        return struct(action = 'redo')        

    elif action.tag == 'HistEdit':
        
        edit = action.contents

        if edit.tag == 'ConfirmDetectionEdit':


            return struct(action='confirm')    
        elif edit.tag == 'TransformPartsEdit':

            transform, ids =  edit.contents
            s, t = transform

            edit = struct(action='transform', type = 'translate', ids = ids)

            corner = all([len(part) == 1 for part in ids.values()])
            if corner:
                edit = edit._extend(type='drag corner')

            if s != 1:
                edit = edit._extend(type='scale')

            return edit
            
        elif edit.tag == 'AddEdit':

            return struct(action='add')    
        elif edit.tag == 'DeletePartsEdit':

            return struct(action='delete', ids = edit.contents)
        elif edit.tag == 'DetectionEdit':

            return struct(action='detection')
        else:
            assert False, "unknown edit type: " + edit.tag            
    
    else:
        assert False, "unknown action type: " + action.tag
    
    



def extract_session(history):

    sessions = []

    def new(t, detections = []):
        return struct(start = t, detections = detections, actions = [])

    def last_time(session):
        if len(session.actions) > 0:
            return session.actions[-1].time 
        else:
            return 0

    open = None
    detections = None

    for (datestr, action) in history:
        t = date.parse(datestr)

            
        if action.tag == 'HistOpenDetections':

            detections = action.contents

            detections = annotate.decode_detections(action.contents.contents, annotate.class_mapping(config))
            open = new(t, detections)

        elif action.tag  == 'HistClose':
            time = (t - open.start).total_seconds()

            if open and len(open.actions) > 0:
                return open._extend(duration = time)

            open = None
        else:
            if open:
                time = (t - open.start).total_seconds()
                duration = time - last_time(open) 

                entry = decode_action(action)._extend(time = time, duration = duration)
                open.actions.append(entry)

    return None


def decode_image(data, config):
    image = annotate.decode_image(data, config=config)
    history = list(reversed(data.history))


    session = extract_session(history)
    
    if session.detections:

        test = struct(prediction = session.detections._sort_on('confidence'), target = image.target)
        compute_mAP = evaluate.mAP_classes([test], num_classes = len(config.classes))

        pr = compute_mAP(0.5)
    print(session.detections)

    return image._extend(detections = session.detections, session = session)


def decode_dataset(data):
    data = to_structs(data)

    config = data.config
    classes = [struct(id = int(k), name = v) for k, v in config.classes.items()]

    images = [decode_image(i, config) for i in data.images if len(i.history) > 0 and i.category != "New" ]


def load_dataset(filename):
    with open(filename, "r") as file:
        str = file.read()
        return decode_dataset(json.loads(str))
    raise Exception('load_file: file not readable ' + filename)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process history dump.')
    parser.add_argument('--input', type=str, help = 'input json file')

    args = parser.parse_args()

    load_dataset(args.input)