
import json
from dataset import annotate
from os import path

import argparse

from tools import struct, to_structs, filter_none, drop_while, concat_lists, map_dict, sum_list, pluck, count_dict, partition_by
from detection import evaluate

from scripts.datasets import quantiles

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

    elif action.tag == 'threshold':
        return struct(action = 'threshold', value=action.contents)        

    elif action.tag == 'edit':
        edit = action.contents

        if edit.tag == 'confirm_detection':
            # confirms = [i for i, b in edit.contents.items() if b == True]
            # return struct(action='confirm' if len(confirms) > 0 else 'select', ids = list(edit.contents.keys()))
            return struct(action='confirm', ids = list(edit.contents.keys()))

        elif edit.tag == 'transform_parts':
            transform, ids =  edit.contents
            s, t = transform

            edit = struct(action='transform', t = 'translate', ids = list(ids.keys()))

            corner = all([len(part) == 1 for part in ids.values()])
            if corner:
                edit = edit._extend(type='drag corner')


            if s != 1:
                edit = edit._extend(type='scale')
            return edit
        elif edit.tag == 'add':
            
            return struct(action='add')
        elif edit.tag == 'delete_parts':

            return struct(action='delete', ids = list(edit.contents.keys()))


        elif edit.tag == 'clear_all':
            return struct(action='delete')

        elif edit.tag == 'set_class':
            class_id, ids = edit.contents
            return struct(action='set_class', ids = ids, class_id = class_id)

        else:
            assert False, "unknown edit type: " + edit.tag

    else:
        assert False, "unknown action type: " + action.tag



def extract_sessions(history, config):

    sessions = []
    def new(t, type, detections = None):
        return struct(start = t, detections = detections, actions = [], type = type)

    def last_time(session):
        if len(session.actions) > 0:
            return session.actions[-1].time
        else:
            return 0

    open = None

    def append_action(action, t):
        nonlocal open
        if open is None:
            open = new(t, None, 'open_new')

        time = (t - open.start).total_seconds()
        previous = open.actions[-1] if len(open.actions) > 0 else None

        # if len(open.actions) > 0:
        #     previous = open.actions[-1]


        if previous and (action.action == 'delete' or action.action == 'transform'):
            if previous.action == 'select' and action.ids == previous.ids:
                time = previous.time 
                open.actions.pop()
              
       
        duration = time - last_time(open)

        if previous is not None and (action.action == previous.action and action.get('ids') == previous.get('ids')):
            return

        entry = action._extend(time = time, duration = min(20, duration))
        open.actions.append(entry)

    for (datestr, action) in history:
        t = date.parse(datestr)

        if action.tag == action.tag == "open":
            assert open is None, "open without close!"

            open = action.contents.open_type
            detections = None

            if open.tag == 'new':
                detections = annotate.decode_detections(open.contents.instances, annotate.class_mapping(config)) 
                
            open = new(t, open.tag, detections)

        elif action.tag  == 'close':
            if open is not None:

                append_action(struct(action = 'submit'), t)            
                sessions.append(open._extend(duration = sum(pluck('duration', open.actions), 0)))

            open = None
        else:
            # assert open is not None, "close without open!"
            append_action(decode_action(action), t)

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


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def image_summaries(history):
    return [image_summary(image) for image in history]


def action_histogram(history, n_splits=None):
    summaries = [image_summary(image) for image in history]

    if n_splits is not None:
        summaries = list(map(sum_list, split(summaries, n_splits)))

    def f(totals):
        return count_dict(pluck('action', totals.actions))

    return list(map(f, summaries))

def image_summaries(history):
    return [image_summary(image) for image in history]

def image_summary(image):
    action_durations = map(lambda action: action.duration, image.actions)
    
    # if len(image.actions) > 0:
    #     print(image.duration, sum(action_durations))

    return struct (
        actions = image.actions,
        n_actions = len(image.actions), 
        duration = image.duration,
        instances = image.target._size
    )


def history_summary(history):
    
    summaries = [image_summary(image) for image in history]
    totals = sum_list(summaries)
    n = len(history)

    actions_count = count_dict(pluck('action', totals.actions))
    # durations = partition_by(totals.actions, lambda action: (action.action, action.duration))

    return struct (

        action_durations = quantiles(pluck('duration', totals.actions)),
        image_durations = quantiles(pluck('duration', summaries)),

        n_actions = quantiles(pluck('n_actions', summaries)),
        actions_count = actions_count
    )

# def summary_actions(session):

#     edited = {}
#     deletes = 0
#     adds = 0
#     confirms = 0
    
#     for action in session.actions:       
#         if action.type == 'delete_parts':
#             deletes += len(action.ids)

#             for i in action.ids:
#                 if edited.get(key): edited.pop(key)

#         if action.type == 'add':
#             adds += 1

#         if action.type == 'confirm':
#             confirms += len(action.confirms)

        


def extract_image(image, config):
    target = annotate.decode_image(image, config).target

    history = image.history
    history = list(reversed(history))

    sessions = extract_sessions(history, config)
   
    if len(sessions) > 0:
        session = join_history(sessions)      

        return struct (
            filename = image.image_file,
            start = session.start,
            detections = sessions[0].detections,
            duration = session.duration,
            actions = session.actions,
            target = target)
        

def extract_histories(dataset):
    images = [extract_image(image, dataset.config) for image in dataset.images]
    images = sorted(filter_none(images), key = lambda image: image.start)

    return images


# def extract_replay(dataset):

    
    
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

