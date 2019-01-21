
import json
from dataset import annotate
from os import path

import argparse

from tools import struct, to_structs, filter_none, drop_while, concat_lists, map_dict
from detection import evaluate

from evaluate import compute_AP

import dateutil.parser as date

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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





def extract_sessions(history, config):

    sessions = []

    def new(t, detections = None):
        return struct(start = t, detections = detections, actions = [])

    def last_time(session):
        if len(session.actions) > 0:
            return session.actions[-1].time
        else:
            return 0

    open = None

    for (datestr, action) in history:
        t = date.parse(datestr)

        if action.tag == 'HistDetections':

            detections = action.contents
            detections = annotate.decode_detections(action.contents, annotate.class_mapping(config))
            open = new(t, detections)

        elif action.tag == 'HistOpen' or action.tag == 'HistReview':
            open = new(t)


        elif action.tag  == 'HistClose':
            time = (t - open.start).total_seconds()

            assert open is not None, 'close without open!'
            sessions.append(open._extend(duration = time))

            open = None
        else:
            assert open is not None, 'edit when not open!'
            time = (t - open.start).total_seconds()
            duration = time - last_time(open)

            entry = decode_action(action)._extend(time = time, duration = duration)
            open.actions.append(entry)

    return sessions


def join_history(sessions):
    duration = 0
    actions = []

    for s in sessions:
        for a in s.actions:
            actions.append(a._extend(time = a.time + duration))

        duration += s.duration

    return struct(start = sessions[0].start, actions = actions, duration = duration, detections = sessions[0].detections)





def decode_image(data, config):
    image = annotate.decode_image(data, config=config)
    history = list(reversed(data.history))

    sessions = extract_sessions(history, config)

    # Find first session with detections
    sessions = drop_while(lambda session: session.detections is None, sessions)

    if len(sessions) > 0:

        detections = sessions[0].detections
        session = join_history(sessions)

        test = struct(prediction = session.detections._sort_on('confidence'), target = image.target)
        compute_mAP = evaluate.mAP_classes([test], num_classes = len(config.classes))

        pr = compute_mAP(0.5)

        return image._extend(session = session, history = history, mAP50 = pr.total.mAP)


def decode_dataset(data):
    data = to_structs(data)

    config = data.config
    classes = [struct(id = int(k), name = v) for k, v in config.classes.items()]

    images = filter_none([decode_image(i, config) for i in data.images])
    images.sort(key = lambda image: image.session.start)

    return struct(classes = classes, images = images)


def load_dataset(filename):
    with open(filename, "r") as file:
        str = file.read()
        return decode_dataset(json.loads(str))
    raise Exception('load_file: file not readable ' + filename)



def plot_maps(series, filename ):


    with PdfPages(path.join(base_path, filename)) as pdf:
        fig = plt.figure()
        fig, ax = plt.subplots(1,1)

        n = max([len(values) for  _, values in series.items()])

        x = list(range(1, n + 1))
        #ax.set_xticks(x)

        ax.set_xlabel("image annotated")
        ax.set_ylabel("mAP@0.5")

        for name, values in series.items():
            plt.plot(x, list(values), label=name)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        fig.suptitle('per image mAP (detected vs annotated)')
        pdf.savefig(fig)


base_path = '/home/oliver/sync/annotate/tests/'
history_logs = {
    #'penguins' : 'penguins.json',
    'scallops' : 'scallops.json'
    }

if __name__ == '__main__':
    runs = {name:load_dataset(path.join(base_path, input)) for name, input in history_logs.items()}

    def map_series(f):
        return {name : list(map(f, run.images)) for name, run in runs.items()}

    # print(runs['penguins'].images[2].session.actions)
    # print(runs['penguins'].images[28].session.actions)

    n = map_series(lambda i: struct(num_actions = len(i.session.actions), 
        n = i.target._size, mAP = i.mAP50, duration = i.session.duration, id = i.id))
    #penguins = n['scallops']

    # for i, s in enumerate(penguins):
    #     print(i, s)


    plot_maps(map_series(lambda i: i.session.duration), filename = 'figures/depth.pdf')
