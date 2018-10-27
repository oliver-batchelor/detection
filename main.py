
import time
import os
import json
import copy
import random

import torch
import torch.optim as optim

from json.decoder import JSONDecodeError

from dataset.annotate import decode_dataset, split_tagged, tagged, decode_image, load_dataset
from detection.models import models
from detection.loss import total_bce

from tools.model import tools

from tools.parameters import default_parameters
from tools import Struct

import connection
import trainer
import evaluate
import math

import arguments
import pprint

pp = pprint.PrettyPrinter(indent=2)


def ready(*xs):
    return all(v is not None for v in xs)


class Reset(Exception):
    def __init__(self, env):
        self.env = env


class NotFound(Exception):
    def __init__(self, filename):
        self.filename = filename


def show_differences(d1, d2, prefix=""):
    unequal_keys = []
    unequal_keys.extend(set(d1.keys()).symmetric_difference(set(d2.keys())))
    for k in d1.keys():
        if d1.get(k, '-') != d2.get(k, '-'):
            unequal_keys.append(k)
    if unequal_keys:
        for k in set(unequal_keys):
            v1 = d1.get(k, '-')
            v2 = d2.get(k, '-')
            if type(v1) != type(v2):
                v1 = type(v1)
                v2 = type(v2)

            print ("{:20s} {:10s}, {:10s}".format(prefix + k, str(v1), str(v2)))



def load_state(model, info):
    model.load_state_dict(info.state)
    return Struct(model = model, score = info.score, epoch = info.epoch)


def try_load(model_path):
    try:
        return torch.load(model_path)
    except (FileNotFoundError, EOFError):
        pass

def load_checkpoint(model_path, model, model_args, no_load=False):
    loaded = try_load(model_path)
    if not (no_load or loaded is None):
        if loaded.args == model_args:
            print("loaded model dataset parameters match, resuming")

            return load_state(copy.deepcopy(model), loaded.best), load_state(copy.deepcopy(model), loaded.current), True
        else:
            print("loaded model dataset parameters differ")
            show_differences(model_args.__dict__,  loaded.args.__dict__)

    best = Struct (model = copy.deepcopy(model), score = 0.0, epoch = 0)
    current = Struct (model = model, score = 0.0, epoch = 0)

    return best, current, False




def initialise(config, dataset, args):
    data_root = config['root']
    classes = {c['id'] : {'shape':c['name']['shape']} for c in dataset.classes}

    model_args = Struct (
        dataset = Struct(classes = classes, input_channels = 3),
        model   = args.model,
        version = 2
    )

    run = 0

    output_path = os.path.join(data_root, args.run_name, "model.pth")
    model, encoder = tools.create(models, model_args.model, model_args.dataset)

    best, current, resumed = load_checkpoint(output_path, model, model_args, args.no_load)
    model, epoch = current.model, current.epoch

    parameters = model.parameter_groups(args.lr, args.fine_tuning)

    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum)
    return Struct(**locals())


def encode_shape(box, config):
    lower, upper =  box[:2], box[2:]

    if config['shape'] == 'CircleConfig':

        centre = ((lower + upper) * 0.5).tolist()
        radius = ((upper - lower).sum().item() / 4)

        circle_shape = {'centre':  centre, 'radius': radius}
        return tagged('CircleShape', circle_shape)

    elif config['shape'] == 'BoxConfig':
        return tagged('BoxShape', {'lower':lower.tolist(), 'upper':upper.tolist()})

    assert False, "unsupported shape config: " + config['shape']


def evaluate_image(env, image, nms_params, device):
    model = env.best.model
    boxes, labels, confs = evaluate.evaluate_image(model.to(device), image, env.encoder, nms_params, device)

    n = len(boxes)
    assert n == len(labels) and n == len(confs)

    classes = env.dataset.classes

    def detection(args):
        box, label, conf = args

        object_class = classes[label.item()]
        config = object_class['name']

        return {
            'shape' : encode_shape(box.cpu(), config),
            'label'  : object_class['id'],
            'confidence' : conf.item()
        }

    return list(map(detection, zip(boxes, labels, confs)))

def detect_request(env, file, nms_params, device):
    path = os.path.join(env.data_root, file)

    if not os.path.isfile(path):
        raise NotFound(file)

    image = env.dataset.load_testing(path, env.args)
    return evaluate_image(env, image, nms_params, device)




def log_lerp(range, t):
    begin, end = range
    return math.exp(math.log(begin) * (1 - t) + math.log(end) * t)

def adjust_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        modified = lr * param_group['modifier'] if 'modifier' in param_group else lr
        param_group['lr'] = modified




def run_trainer(args, conn = None, env = None):

    device = torch.cuda.current_device()


    def send_command(command, data):

        if conn is not None:
            str = json.dumps(tagged(command, data))
            conn.send(str)

    def process_command(str):
        nonlocal env, device

        try:
            tag, data = split_tagged(json.loads(str))
            print("recieved command: " + tag)

            if tag == 'TrainerInit':

                config, dataset = decode_dataset(data)
                env = initialise(config, dataset, args)


            elif tag == 'TrainerUpdate':
                file, image_data = data

                image = decode_image(image_data, env.config)
                category = image_data['category']
                print ("updating '" + file + "' in " + category)

                env.dataset.update_image(file, image, category)


            elif tag == 'TrainerDetect':
                reqId, image, nms_prefs = data

                nms_params = {
                    'nms_threshold'     :   nms_prefs['nms'],
                    'class_threshold'   :   nms_prefs['threshold'],
                    'max_detections'    :   nms_prefs['detections']
                }

                if env is not None:

                    results = detect_request(env, image, nms_params, device)
                    send_command('TrainerDetections', (reqId, image, results, (env.run, env.best.epoch)))

                else:
                    send_command('TrainerReqError', [reqId, image, "model not available yet"])

            else:
                send_command('TrainerError', "unknown command: " + tag)


        except (JSONDecodeError) as err:
            send_command('TrainerError', repr(err))
            return None

    def poll_command():
        if conn and conn.poll():
            cmd = conn.recv()
            process_command(cmd)

    def train_update(n, total):
        lr = log_lerp((args.lr, args.lr * 0.1), n / total) if args.lr > 0 else 0
        adjust_learning_rate(lr, env.optimizer)
        poll_command()

    def test_update(n, total):
        poll_command()

    def training_cycle():
        model = env.model.to(device)
        has_training = len(env.dataset.train_images) > 0

        if has_training:
            print("training {}:".format(env.epoch))
            stats = trainer.train(model, env.dataset.sample_train(args, env.encoder),
                        evaluate.eval_train(total_bce, device), env.optimizer, hook=train_update)
            evaluate.summarize_train("train", stats, env.epoch)

        score = 0
        if len(env.dataset.test_images) > 0:
            nms_params = args.subset('nms_threshold',  'class_threshold', 'max_detections')

            print("testing {}:".format(env.epoch))
            stats = trainer.test(model, env.dataset.test(args),
                evaluate.eval_test(env.encoder, nms_params=nms_params, device=device), hook=test_update)

            score = evaluate.summarize_test("test", stats, env.epoch)

        if has_training:
            is_best = score >= env.best.score
            if is_best:
                env.best = Struct(model = copy.deepcopy(model), score = score, epoch = env.epoch)

            current = Struct(state = model.state_dict(), epoch = env.epoch, score = score)
            best = Struct(state = env.best.model.state_dict(), epoch = env.best.epoch, score = env.best.score)

            save_checkpoint = Struct(current = current, best = best, args = env.model_args, run = env.run)
            torch.save(save_checkpoint, env.output_path)

            send_command("TrainerCheckpoint", ((env.run, env.epoch), score, is_best))
            env.epoch = env.epoch + 1


    while(True):
        if env is not None:
            training_cycle()
        poll_command()


def run_main():
    args = arguments.get_arguments()
    pp.pprint(args.to_dicts())

    p, conn = None, None
    env = None

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.remote:
        print("connecting to: " + args.remote)
        p, conn = connection.connect('ws://' + args.remote)

    if args.input:
        print("loading from: " + args.input)

        config, dataset = load_dataset(args.input)

        env = initialise(config, dataset, args)

    try:
        run_trainer(args, conn, env=env)
    except (KeyboardInterrupt, SystemExit):
        p.terminate()


if __name__ == '__main__':
    run_main()
