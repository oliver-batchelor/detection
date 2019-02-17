
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

    if action.tag == 'undo':
        return struct(action = 'undo')

    elif action.tag == 'redo':
        return struct(action = 'redo')

    elif action.tag == 'edit':

        edit = action.contents

        if edit.tag == 'confirm_detection':


            return struct(action='confirm')
        elif edit.tag == 'transform_parts':

            transform, ids =  edit.contents
            s, t = transform

            edit = struct(action='transform', type = 'translate', ids = ids)

            corner = all([len(part) == 1 for part in ids.values()])
            if corner:
                edit = edit._extend(type='drag corner')

            if s != 1:
                edit = edit._extend(type='scale')

            return edit

        elif edit.tag == 'add':

            return struct(action='add')
        elif edit.tag == 'delete_parts':

            return struct(action='delete', ids = edit.contents)
        elif edit.tag == 'detection':

            return struct(action='detection')
        else:
            assert False, "unknown edit type: " + edit.tag

    else:
        assert False, "unknown action type: " + action.tag





def extract_sessions(history, config):

    sessions = []

    def new(t, detections = None):
        return struct(start = t, detections = None, actions = [])

    def last_time(session):
        if len(session.actions) > 0:
            return session.actions[-1].time
        else:
            return 0

    open = None

    for (datestr, action) in history:
        t = date.parse(datestr)

        if open is None:
            open = new(t)
    
        if action.tag == 'open_new' or action.tag == 'open_review':
            detections = action.contents
            detections = annotate.decode_detections(action.contents, annotate.class_mapping(config))
            open.detections = detections

        elif action.tag  == 'close':
            if len(open.actions) > 0:
                time = (t - open.start).total_seconds()

                entry = struct(action = 'close')._extend(time = time, duration = duration)
                open.actions.append(entry)
                sessions.append(open._extend(duration = time))

            open = None
        else:
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

    return struct(start = sessions[0].start, actions = actions, duration = duration)


def action_durations(actions):
    return [action.duration for action in actions if action.duration > 0]


def history_summary(history):



def extract_image(image, config):
    target = annotate.decode_image(image).target

    history = image.history
    history = list(reversed(history))

    sessions = extract_sessions(history, config)
    # sessions = drop_while(lambda session: session.detections is None, sessions)

    if len(sessions) > 0):
        return struct (
            filename = image.filename,
            start = sessions[0].start,
            target = target,
            sessions = sessions)
        


    
        # detections = sessions[0].detections
        # session = join_history(sessions)

        # test = struct(prediction = session.detections._sort_on('confidence'), target = image.target)    
        # compute_mAP = evaluate.mAP_classes([test], num_classes = len(dataset.config.classes))

        # pr = compute_mAP(0.5)        

        # return struct(
        #     filename = image.filename,
        #     num_actions = len(session.actions), duration = session.duration, 
        #     start = session.start,
        #     durations = action_durations(actions),
        #     mAP50 = pr.total.mAP)
            

def extract_histories(dataset):
    images = [extract_image(image.history, dataset.config) for image in  dataset.images]

    return images


# def summary


# def plot_maps(series, filename, title='title'):

#     with PdfPages(path.join(base_path, filename)) as pdf:
#         fig = plt.figure()
#         fig, ax = plt.subplots(1,1)

#         ax.set_xlabel("time")
#         ax.set_ylabel("annotations")

#         for name, values in series.items():
#             xs, ys = zip(*values)
#             plt.plot(xs, ys, label=name)

#         handles, labels = ax.get_legend_handles_labels()
#         ax.legend(handles, labels)

#         fig.suptitle(title)
#         pdf.savefig(fig)

