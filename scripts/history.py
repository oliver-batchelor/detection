
import json
from dataset import annotate
from os import path

import argparse

from tools import table, struct, to_structs, filter_none, drop_while, \
     concat_lists, map_dict, sum_list, pluck, count_dict, partition_by, show_shapes, Struct

from detection import evaluate

from collections import deque

import tools

from scripts.datasets import quantiles

from evaluate import compute_AP

import dateutil.parser as date

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch


def decode_action(action):
    if action.tag == 'undo':
        return struct(action = 'undo')

    elif action.tag == 'redo':
        return struct(action = 'redo')

    elif action.tag == 'threshold':
        return struct(action = 'threshold', value=action.contents)        

    elif action.tag == 'close':
        return struct(action='submit')

    elif action.tag == 'edit':
        edit = action.contents

        if edit.tag == 'confirm_detection':
            return struct(action='confirm', ids = list(edit.contents.keys()))

        elif edit.tag == 'transform_parts':
            transform, ids =  edit.contents
            s, t = transform

            return struct(action='transform', t = 'translate', ids = list(ids.keys()))
        elif edit.tag == 'add':
            
            return struct(action='add')
        elif edit.tag == 'delete_parts':

            ids = list(edit.contents.keys())

            return struct(action='delete', ids = ids)


        elif edit.tag == 'clear_all':

            return struct(action='delete')

        elif edit.tag == 'set_class':
            class_id, ids = edit.contents
            return struct(action='set_class', ids = ids, class_id = class_id)
        
        else:
            assert False, "unknown edit type: " + edit.tag


    else:
        assert False, "unknown action type: " + action.tag


def extract_session(session, config):

    start = date.parse(session.time)
    detections = []
    actions = []

  
    detections = annotate.decode_detections(session.open.contents.instances, annotate.class_mapping(config)) \
        if session.open.tag == "new" else empty_detections

    def previous():
        return actions[-1] if len(actions) > 0 else None

    def previous_time():        
        return (actions[-1].time if len(actions) > 0 else 0)

    for (datestr, action) in session.history:
        t = date.parse(datestr)
        action = decode_action(action)

        prev = previous()
        if prev and action.action in ['transform', 'delete']:
            if prev.action == 'confirm' and prev.ids == action.ids:
                actions.pop()

        time = (t - start).total_seconds()    
        duration = time - previous_time()
            
        actions.append(action._extend(time = time, duration = min(60, duration)))        

    duration = sum (pluck('duration', actions))
    end = actions[-1].time

    return struct(start = start, detections = detections, actions = actions, \
        duration = duration, real_duration = end,  type = session.open.tag, threshold=session.threshold)


def action_durations(actions):
    return [action.duration for action in actions if action.duration > 0]


def image_summaries(history):
    return [image_summary(image) for image in history]


def instance_windows(history, window=100):
    windows = []

    for i in range(len(history)):
        images = []
        n = 0 
        
        def add(k):
            nonlocal n

            if n < window and k >= 0 and k < len(history):
                images.append(k)
                n = n + history[k].target._size

        j = 1
        add(i)
        while n < window:
            add(i + j)
            add(i - j)
            
            j = j + 1

        windows.append(images)

    return windows



empty_detections = table (
        bbox = torch.FloatTensor(0, 4),
        label = torch.LongTensor(0),
        confidence = torch.FloatTensor(0))

def image_result(image):      
    prediction = empty_detections if image.detections is None else image.detections
    return struct(target = image.target, prediction = prediction )

def running_AP(history, window=100):
    windows = instance_windows(history, window)
        
    image_pairs =  filter_none([image_result(image) for image in history])
    mAPs = [evaluate.mAP_subset(image_pairs, iou=t/100) for t in list(range(50, 100, 5))]

    def compute_AP(w):
        return sum([mAP(w).mAP for mAP in mAPs]) / 10

    return [compute_AP(w) for w in windows]


def running_mAP(history, window=100, iou=0.5):
    windows = instance_windows(history, window)
    #windows = [[i] for i in range(len(history))]
    
    image_pairs =  filter_none([image_result(image) for image in history])
    mAP = evaluate.mAP_subset(image_pairs, iou=iou)

    return [mAP(w).mAP for w in windows]


def image_summaries(history):
    return [image_summary(image) for image in history]

annotation_types = ['positive', 'weak positive', 'modified positive', 'false negative', 'false positive']

def annotation_categories(image):
    mapping = {'add':'false negative', 'confirm':'weak positive', 'detect':'positive'}
    t = image.threshold

    def get_category(s):
        if s.status.tag == "active":
            if s.created_by.tag == "detect":
                return "modified positive" if (s.changed_class or s.transformed) else "positive"
            return mapping.get(s.created_by.tag)

        if s.created_by.tag == "detect" and s.status.tag == "deleted":
            detection = s.created_by.contents
            if detection.confidence >= t:
                return "false positive"

    created = filter_none([get_category(s) for s in image.ann_summaries])
    return count_struct(created, annotation_types)

def image_summary(image):
    action_durations = map(lambda action: action.duration, image.actions)
    
    
    # if len(image.actions) > 0:
    #     print(image.duration, sum(action_durations))

    return struct (
        actions = image.actions,
        n_actions = len(image.actions), 
        duration = image.duration,
        real_duration = image.real_duration,
        instances = image.target._size,
        annotation_types = annotation_categories(image)
    )

def count_struct(values, keys):
    d = count_dict(values)
    zeroes = Struct({k:0 for k in keys})

    return zeroes._extend(**d)



def history_summary(history):
    
    summaries = [image_summary(image) for image in history]
    totals = sum_list(summaries)
    n = len(history)

    actions_count = count_dict(pluck('action', totals.actions))
    durations = pluck('duration', summaries)
    real_durations = pluck('real_duration', summaries)

    total_actions = sum(actions_count.values(), 0)
    total_duration = sum(durations, 0)

    instances = sum(pluck('instances', summaries), 0)


    # durations = partition_by(totals.actions, lambda action: (action.action, action.duration))

    return summaries, struct (
        
        action_durations = quantiles(pluck('duration', totals.actions)),
        image_durations = quantiles(durations),

        n_actions = quantiles(pluck('n_actions', summaries)),
        instances_image = quantiles(pluck('instances', summaries)),

        annotation_types = totals.annotation_types,

        actions_count = actions_count,

        total_minutes = total_duration / 60,
        total_actions = total_actions,

        real_minutes = sum(real_durations, 0),

        actions_minute      = 60 * total_actions / total_duration,
        instances_minute    = 60 * instances / total_duration
    )
        

def extract_image(image, config):
    target = annotate.decode_image(image, config).target

    if len(image.sessions) > 0:          
        session = extract_session(image.sessions[0], config)
    
        return struct (
            filename = image.image_file,
            start = session.start,
            detections = session.detections,
            duration = session.duration,
            real_duration = session.real_duration,
            actions = session.actions,
            threshold = session.threshold,
            ann_summaries = image.summaries,
            target = target)
        

def extract_histories(dataset):
    images = [extract_image(image, dataset.config) for image in dataset.images]
    images = sorted(filter_none(images), key = lambda image: image.start)

    return images
