
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


def show_differences(d1, d2, desc1, desc2):
    unequal_keys = []
    unequal_keys.extend(set(d1.keys()).symmetric_difference(set(d2.keys())))
    for k in d1.keys():
        if d1.get(k, '-') != d2.get(k, '-'):
            unequal_keys.append(k)
    if unequal_keys:
        print ('key', '\t' + desc1, desc2)
        for k in set(unequal_keys):
            print (str(k)+'\t'+d1.get(k, '-')+'\t '+d2.get(k, '-'))


def try_load(model_path, model, model_args, no_load=False):
    best = Struct (model = copy.deepcopy(model), score = 0.0, epoch = 0)
    current = Struct (model = model, score = 0.0, epoch = 0)

    if not no_load:
        try:
            loaded = torch.load(model_path)
            if loaded.args.dataset == model_args.dataset:
                print("loaded model dataset parameters match, resuming")
                return loaded.best, loaded.current, model_args
            else:
                print("loaded model dataset parameters differ")
                show_differences(model_args.dataset.__dict__,  loaded_args.dataset.__dict__, "args", "loaded")


        except (FileNotFoundError, AttributeError):
            pass


    return best, current, model_args




def initialise(config, dataset, args):
    data_root = config['root']
    classes = {c['id'] : {'shape':c['name']['shape']} for c in dataset.classes}

    model_args = Struct (
        dataset = Struct(classes = classes, input_channels = 3),
        model   = args.model,
        version = 1
    )

    epoch = 0

    output_path = os.path.join(data_root, args.run_name, "model.pth")
    model, encoder = tools.create(models, model_args.model, model_args.dataset)

    best, current, model_args = try_load(output_path, model, model_args, args.no_load)
    model, epoch = current.model, current.epoch


    parameters = model.parameter_groups(args.lr, args.fine_tuning)


    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum)
    return Struct(**locals())


def encode_box(box):
    lower, upper =  box[:2].tolist(), box[2:].tolist()
    return {'lower':lower, 'upper':upper}


def evaluate_image(env, image, nms_params, device):
    env.best_model.to(device)
    boxes, labels, confs = evaluate.evaluate_image(env.best_model, image, env.encoder, nms_params, device)

    n = len(boxes)
    assert n == len(labels) and n == len(confs)

    classes = env.dataset.classes

    def detection(args):
        box, label, conf = args
        return {
            'bounds' : encode_box(box.cpu()),
            'label'  : classes[label.item()]['id'],
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
        str = json.dumps(tagged(command, data))
        conn.send(str)


    def process_command(str):
        nonlocal env, device

        try:
            tag, data = split_tagged(json.loads(str))
            print("recieved command: " + tag)

            if tag == 'TrainerDataset':

                config, dataset = decode_dataset(data)
                env = initialise(config, dataset, args)

                raise Reset(env)

            elif tag == 'TrainerUpdate':
                file, image_data = data

                image = decode_image(image_data, env.config)
                category = image_data['category']
                print ("updating '" + file + "' in " + category)

                env.dataset.update_image(file, image, category)


            elif tag == 'TrainerDetect':
                clientId, image, nms_prefs = data

                nms_params = {
                    'nms_threshold'     :   nms_prefs['nms'],
                    'class_threshold'   :   nms_prefs['threshold'],
                    'max_detections'    :   nms_prefs['detections']
                }

                if env is not None:
                    result = detect_request(env, image, nms_params, device)
                    send_command('TrainerDetections', (clientId, image, result))

                else:
                    send_command('TrainerReqError', [clientId, "model not available yet"])

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

        current = Struct(model = copy.deepcopy(model), epoch = env.epoch, score = score)
        if score >= env.best.score and has_training:
            env.best = current

        checkpoint = Struct(current = current, best = env.best, args = env.model_args)
        torch.save(checkpoint, env.output_path)


        env.epoch = env.epoch + 1


    while(True):
        try:
            if env is not None:
                training_cycle()
            poll_command()

        except Reset as reset:
            env   = reset.env

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
