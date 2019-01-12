
import time
import os
import json
import copy
import random

import sys
import traceback

import torch
from torch import nn
import torch.optim as optim

from json.decoder import JSONDecodeError

from dataset.annotate import decode_dataset, split_tagged, tagged, decode_image, init_dataset
from dataset.imports import load_dataset

from dataset.detection import least_recently_evaluated

from detection.models import models
from detection.loss import focal_loss

from tools.model import tools

from tools.parameters import default_parameters, get_choice
from tools import struct, logger, show_shapes, to_structs, Struct


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


def copy_partial(dest, src):
    assert src.dim() == dest.dim()

    for d in range(0, src.dim()):

        if src.size(d) > dest.size(d):
            src = src.narrow(d, 0, dest.size(d))
        else:
            dest = dest.narrow(d, 0, src.size(d))

    dest.copy_(src)

def load_state_partial(model, src):
    dest = model.state_dict()

    for k, dest_param in dest.items():
        if k in src:
            source_param = src[k]

            if source_param.dim() == dest_param.dim():
                copy_partial(dest_param, source_param)




def load_state(model, info):
    load_state_partial(model, info.state)
    return struct(model = model, score = info.score, epoch = info.epoch)

def new_state(model):
    return struct (model = model, score = 0.0, epoch = 0)

def try_load(model_path):
    try:
        return torch.load(model_path)
    except (FileNotFoundError, EOFError, RuntimeError):
        pass

def load_model(model_path):
    loaded = try_load(model_path)
    assert loaded is not None, "failed to load model from " + model_path
 
    args = loaded.args


    model, encoder = tools.create(models, args.model, args.dataset)
    best = load_state(model, loaded.best)

    return model, encoder, args


def load_checkpoint(model_path, model, model_args, args):
    loaded = try_load(model_path)

    if not (args.no_load or not (type(loaded) is Struct)):

        current = load_state(model, loaded.best if args.restore_best else loaded.current)
        best = load_state(copy.deepcopy(model), loaded.best)

        print(loaded.args)

        if loaded.args == model_args:
            print("loaded model dataset parameters match, resuming")

        else:
            print("loaded model dataset parameters differ, loading partial")
            show_differences(model_args.__dict__,  loaded.args.__dict__)

            best.score = 0.0
            best.epoch = current.epoch

        return best, current, True

    return new_state(copy.deepcopy(model)), new_state(model), False




def initialise(config, dataset, args):
    data_root = config.root

    model_args = struct (
        dataset = struct(
            classes = dataset.classes,
            input_channels = 3),
        model   = args.model,
        version = 2
    )

    run = 0

    debug = struct(
        predictions = args.debug_predictions or args.debug_all,
        boxes = args.debug_boxes  or args.debug_all
    )

    output_path, log = logger.make_experiment(data_root, args.run_name, load=not args.no_load, dry_run=args.dry_run)
    model_path = os.path.join(output_path, "model.pth")

    model, encoder = tools.create(models, model_args.model, model_args.dataset)

    set_bn_momentum(model, args.bn_momentum)

    best, current, resumed = load_checkpoint(model_path, model, model_args, args)
    model, epoch = current.model, current.epoch

    running_average = [flatten_parameters(best.model).cpu()] if epoch >= args.average_start else []

    parameters = model.parameter_groups(args.lr, args.fine_tuning)

    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum)

    return struct(**locals())


def encode_shape(box, config):
    lower, upper =  box[:2], box[2:]

    if config.shape == 'circle':

        centre = ((lower + upper) * 0.5).tolist()
        radius = ((upper - lower).sum().item() / 4)

        circle_shape = struct(centre = centre, radius = radius)
        return tagged('circle', circle_shape)

    elif config.shape == 'box':
        return tagged('box', struct (lower = lower.tolist(), upper = upper.tolist()))

    assert False, "unsupported shape config: " + config.shape


def make_detections(env, prediction):
    classes = env.dataset.classes

    def detection(p):
        object_class = classes[p.label]
        config = object_class.name

        return struct (
            shape      =  encode_shape(p.bbox.cpu(), config),
            label      =  object_class.id,
            confidence = p.confidence.item()
        )

    return struct(detections = list(map(detection, prediction._sequence())), networkId = (env.run, env.best.epoch))

def evaluate_image(env, image, nms_params, device):
    model = env.best.model
    prediction = evaluate.evaluate_image(model.to(device), image, env.encoder, nms_params, device)

    return make_detections(env, prediction)


def detect_request(env, file, nms_params, device):
    path = os.path.join(env.data_root, file)

    if not os.path.isfile(path):
        raise NotFound(file)

    image = env.dataset.load_inference(path, env.args)
    return evaluate_image(env, image, nms_params, device)


def log_lerp(range, t):
    begin, end = range
    return math.exp(math.log(begin) * (1 - t) + math.log(end) * t)

def adjust_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        modified = lr * param_group['modifier'] if 'modifier' in param_group else lr
        param_group['lr'] = modified


def flatten_parameters(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()], 0)

def load_flattened(model, flattened):
    offset = 0
    for param in model.parameters():

        data = flattened[offset:offset + param.nelement()].view(param.size())
        param.data.copy_(data)
        offset += param.nelement()

    return model


def append_window(x, xs, window):
    return [x, *xs][:window]


def set_bn_momentum(model, mom):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = mom

def get_nms_params(args):
    return struct(
            nms = args.nms_threshold,
            threshold = args.class_threshold,
            detections = args.max_detections)


def run_testing(model, env, device=torch.cuda.current_device(), hook=None):
    score = 0
    if len(env.dataset.test_images) > 0:
        print("testing {}:".format(env.epoch))
        test_stats = trainer.test(env.dataset.test(env.args),
            evaluate.eval_test(model.eval(), env.encoder, nms_params=get_nms_params(env.args), device=device), hook=hook)

        score = evaluate.summarize_test("test", test_stats, env.dataset.classes, env.epoch, log=env.log)
    return score


def run_detections(model, env, device=torch.cuda.current_device(), hook=None, n=None):
    images = least_recently_evaluated(env.dataset.new_images, n = n)

    if len(images) > 0:
        results = trainer.test(env.dataset.test_on(images, env.args),
                evaluate.eval_test(model.eval(), env.encoder, nms_params=get_nms_params(env.args), device=device), hook=hook)    


        return {result.file[0] : make_detections(env, result.prediction) for result in results}

    #trainer.evaluate(env.dataset.test(env.args))
    # run_detections



class UserCommand(Exception):
    def __init__(self, command):
        super(UserCommand, self).__init__("")
        self.command = command

def run_trainer(args, conn = None, env = None):

    device = torch.cuda.current_device()
    def send_command(command, data):

        if conn is not None:
            str = json.dumps(tagged(command, data)._to_dicts())           
            conn.send(str)

    def process_command(str):
        nonlocal env, device

        if str is None:
            print("Server disconnected.")
            raise UserCommand('pause')

        try:
            tag, data = split_tagged(to_structs(json.loads(str)))
            if tag == 'command':
                raise UserCommand(data)


            elif tag == 'init':
                config, dataset = init_dataset(data)
                env = initialise(config, dataset, args)

                raise UserCommand('resume')


            elif tag == 'update':
                file, image_data = data

                image = decode_image(image_data, env.config)
                env.dataset.update_image(image)


            elif tag == 'detect':
                reqId, image, nms_params = data

                if env is not None:

                    results = detect_request(env, image, nms_params, device)
                    send_command('detections', (reqId, image, results))

                else:
                    send_command('req_error', [reqId, image, "model not available yet"])

            else:
                send_command('error', "unknown command: " + tag)


        except (JSONDecodeError) as err:
            send_command('error', repr(err))
            return None

    def poll_command():
        while conn and conn.poll():
            cmd = conn.recv()
            process_command(cmd)

    def train_update(n, total):
        lr = log_lerp((args.lr, args.lr * args.lr_epoch_decay), n / total) if args.lr > 0 else 0

        adjust_learning_rate(lr, env.optimizer)

        activity = struct(tag = 'train', epoch = env.epoch)
        send_command('progress', struct(activity = activity, progress = (n, total)))
        
        poll_command()


        
    def update(name):
        def f(n, total):
            activity = struct(tag = name, epoch = env.epoch)
            send_command('progress', struct(activity = activity, progress = (n, total)))
            poll_command()
        return f

    def training_cycle():
        if env == None or len(env.dataset.train_images) == 0:
            return None

        env.log.set_step(env.epoch)
        model = env.model.to(device)

        # print("estimating statistics {}:".format(env.epoch))
        # stats = trainer.test(env.dataset.sample_train(args, env.encoder), evaluate.eval_stats(env.dataset.classes, device=device))
        # evaluate.summarize_stats(stats, env.epoch)


        print("training {}:".format(env.epoch))
        train_stats = trainer.train(env.dataset.sample_train(args, env.encoder),
                    evaluate.eval_train(model.train(), focal_loss, env.debug, device=device), env.optimizer, hook=train_update)
        evaluate.summarize_train("train", train_stats, env.dataset.classes, env.epoch, log=env.log)

        # Save parameters for model averaging
        training_params = flatten_parameters(model).cpu()

        is_averaging = env.epoch >= args.average_start and args.average_window > 1
        if is_averaging:
            print("averaging:".format(env.epoch))
            env.running_average = append_window(training_params, env.running_average, args.average_window)

            # Replace model with averaged model for purposes of testing
            load_flattened(model, sum(env.running_average) / len(env.running_average))

            trainer.update_bn(env.dataset.sample_train(args._extend(batch_size = 16, epoch_size = 128), env.encoder),
                evaluate.eval_forward(model.train(), device))

        score = run_testing(model, env, device=device, hook=update('test'))


        is_best = score >= env.best.score
        if is_best:
            env.best = struct(model = copy.deepcopy(model), score = score, epoch = env.epoch)

        if is_averaging:
            load_flattened(model, training_params) # Restore parameters

        current = struct(state = model.state_dict(), epoch = env.epoch, score = score)
        best = struct(state = env.best.model.state_dict(), epoch = env.best.epoch, score = env.best.score)

        save_checkpoint = struct(current = current, best = best, args = env.model_args, run = env.run)
        torch.save(save_checkpoint, env.model_path)

        send_command("checkpoint", ((env.run, env.epoch), score, is_best))
        env.epoch = env.epoch + 1


        if args.detections > 0 and conn is not None:
            results = run_detections(model, env, device=device, hook=update('detect'), n=args.detections)
            send_command('detections_many', results)


        env.log.flush()


    def review_all():
        print("reviewing...")

    def detect_all():
        print("detecting...")
        

    def paused():
        send_command('progress', None)

        while(True):
            poll_command()


    activities = struct(
        detect = detect_all,
        review = review_all,
        resume = training_cycle,
        test = training_cycle,
        pause = paused
    )

    activity = training_cycle

    while(True):
        try:
            activity()
            poll_command()

        except UserCommand as cmd:
            assert cmd.command in activities, "Unknown command " + cmd.command

            print ("User command: ", cmd.command)
            activity = activities[cmd.command]
                

def run_main():
    args = arguments.get_arguments()
    pp.pprint(args._to_dicts())

    p, conn = None, None
    env = None

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    choice, input_args = get_choice(args.input)

    if choice == 'remote':
        print("connecting to: " + input_args.host)
        p, conn = connection.connect('ws://' + input_args.host)
    else:
        config, dataset = load_dataset(args)
        env = initialise(config, dataset, args)


    try:
        run_trainer(args, conn, env=env)
    except (KeyboardInterrupt, SystemExit):
        p.terminate()
    except Exception:
        traceback.print_exc()
        p.terminate()


if __name__ == '__main__':
    run_main()
