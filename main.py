
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

from dataset.annotate import decode_dataset, split_tagged, tagged, decode_image, init_dataset, decode_object_map
from dataset.imports import load_dataset

from dataset.detection import least_recently_evaluated

from detection.models import models
from detection.loss import batch_focal_loss
from detection import box

from tools.model import tools

from tools.parameters import default_parameters, get_choice
from tools import table, struct, logger, show_shapes, to_structs, Struct

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

    pause_time = args.auto_pause
    running_average = [] if epoch >= args.average_start else []

    parameters = model.parameter_groups(args.lr, args.fine_tuning)

    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    
    device = torch.cuda.current_device()
    loss_func = batch_focal_loss 
    
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


def make_detections(env, predictions):
    classes = env.dataset.classes

    def detection(p):
        object_class = classes[p.label]
        config = object_class.name

        return struct (
            shape      =  encode_shape(p.bbox.cpu(), config),
            label      =  object_class.id,
            confidence = p.confidence.item(),
            match = p.match.item() if 'match' in p else None
        )
    detections = list(map(detection, predictions))

    def score(ds):
        return sum(d.confidence ** 2 for d in ds)      

    stats = struct (
        score   = score(detections),
        classes = {c.id : score([d for d in detections if d.label == c.id]) for c in classes}
    )

    return struct(detections = detections, networkId = (env.run, env.best.epoch), stats = stats)



def evaluate_detections(env, image, nms_params):
    model = env.best.model
    detections = evaluate.evaluate_image(model.to(env.device), image, env.encoder, nms_params=nms_params, device=env.device)
    return make_detections(env, list(detections._sequence()))


def select_matching(ious, prediction, threshold = 0.5):
    matching = ious < threshold

    confidence = prediction.confidence.unsqueeze(1).expand(matching.size()).masked_fill(~matching, 0)
    
    _, max_ids = confidence.max(0)
    return prediction._index_select(max_ids)

    # print(predefined)
    # matching = (max_ious > threshold).nonzero().squeeze()
    # print(prediction._index_select (matching)._sort_on('confidence'))

def suppress_boxes(ious, prediction, threshold = 0.5):
    max_ious, _ = ious.max(1)  
    return prediction._extend(
        confidence = prediction.confidence.masked_fill(max_ious > threshold, 0))

def table_list(t):
    return list(t._sequence())

def evaluate_review(env, image, nms_params, review):
    model = env.best.model

    model.eval()
    with torch.no_grad():
        prediction = evaluate.evaluate_decode(model.to(env.device), image, env.encoder, device=env.device)

        ious = box.iou(prediction.bbox, review.bbox.to(env.device))
        review_predictions = select_matching(ious, prediction, threshold = nms_params.threshold)

       
        prediction = suppress_boxes(ious, prediction, threshold = nms_params.threshold)

        

        detections = table_list(review_predictions._extend(match = review.id)) + \
                     table_list(env.encoder.nms(prediction, nms_params=nms_params))

        return make_detections(env, detections)


# def match_predictions(bbox, predictions, threshold=0.5):
#     ious = iou(predictions.bbox, bbox)

#     _, max_ids = ious.max(1)
#     return predictions._index_select(max_ids)

# def review_request(env, file, nms_params, device):


def detect_request(env, file, nms_params, review=None):
    path = os.path.join(env.data_root, file)

    if not os.path.isfile(path):
        raise NotFound(file)

    image = env.dataset.load_inference(file, path, env.args)

    if review is None:
        return evaluate_detections(env, image, nms_params)
    else:
        return evaluate_review(env, image, nms_params, review)


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

def test_images(images, model, env, hook=None):
    eval_test = evaluate.eval_test(model.eval(), env.encoder, loss_func = env.loss_func,
        debug = env.debug, nms_params=get_nms_params(env.args), device=env.device, crop_boxes=env.args.crop_boxes)

    return trainer.test(env.dataset.test_on(images, env.args, env.encoder), eval_test, hook=hook)


def run_testing(name, images, model, env, hook=None):
    score = 0
    if len(images) > 0:
        print("{} {}:".format(name, env.epoch))
        results = test_images(images, model, env,  hook)

        score = evaluate.summarize_test(name, results, env.dataset.classes, env.epoch, log=env.log)
    return score


def run_detections(model, env, hook=None, n=None):
    images = least_recently_evaluated(env.dataset.new_images, n = n)

    if len(images) > 0:
        results = test_images(images, model, env, hook)

        # print(show_shapes(results[0].prediction))
        return {result.id[0] : make_detections(env, table_list(result.prediction)) for result in results}

def add_multimap(m, k, x):
    xs = m[k] if k in m else []
    xs.append(x)
    m[k] = xs


def report_training(results):
    images = {}

    for r in results:
        for file, loss in r.files:
            add_multimap(images, file, struct(loss = loss))

    return images
        

class UserCommand(Exception):
    def __init__(self, command):
        super(UserCommand, self).__init__("")
        self.command = command

def run_trainer(args, conn = None, env = None):

    def send_command(command, data):

        if conn is not None:
            # try:
            command_str = json.dumps(tagged(command, data)._to_dicts())           
            conn.send(command_str)
            # except TypeError:
            #     print("error serialising command: " + str((command, data)))


    def process_command(str):
        nonlocal env

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

                if not args.paused:
                    raise UserCommand('resume')

            elif tag == 'update':
                file, image_data = data

                image = decode_image(image_data, env.config)
                env.dataset.update_image(image)

                if image.category == 'validate':
                    env.best.score = 0

                if env.pause_time == 0:
                    env.pause_time = env.args.auto_pause                    
                    raise UserCommand('resume')
                else:
                    env.pause_time = env.args.auto_pause

            elif tag == 'detect':
                reqId, file, annotations, nms_params = data

                review = decode_object_map(annotations, env.config) if len(annotations) > 0 else None

                if env is not None:
                    results = detect_request(env, file, nms_params, review=review)
                    send_command('detect_request', (reqId, file, results))

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
        model = env.model.to(env.device)

        # print("estimating statistics {}:".format(env.epoch))
        # stats = trainer.test(env.dataset.sample_train(args, env.encoder), evaluate.eval_stats(env.dataset.classes, device=device))
        # evaluate.summarize_stats(stats, env.epoch)


        print("training {}:".format(env.epoch))
        train_stats = trainer.train(env.dataset.sample_train(args, env.encoder),
                    evaluate.eval_train(model.train(), env.loss_func, env.debug, device=env.device), env.optimizer, hook=train_update)
        evaluate.summarize_train("train", train_stats, env.dataset.classes, env.epoch, log=env.log)

        
        send_command('training', report_training(train_stats))

        # Save parameters for model averaging
        training_params = flatten_parameters(model).cpu()

        is_averaging = env.epoch >= args.average_start and args.average_window > 1
        if is_averaging:
            env.running_average = append_window(training_params, env.running_average, args.average_window)

            # Replace model with averaged model for purposes of testing
            load_flattened(model, sum(env.running_average) / len(env.running_average))

            print("updating average batch norm:".format(env.epoch))
            trainer.update_bn(env.dataset.sample_train(args._extend(batch_size = 16, epoch_size = 128), env.encoder),
                evaluate.eval_forward(model.train(), device=env.device))



        score = run_testing('validate', env.dataset.validate_images, model, env,  hook=update('validate'))


        is_best = score >= env.best.score
        if is_best:
            env.best = struct(model = copy.deepcopy(model), score = score, epoch = env.epoch)

        if is_averaging:
            load_flattened(model, training_params) # Restore parameters

        current = struct(state = model.state_dict(), epoch = env.epoch, score = score)
        best = struct(state = env.best.model.state_dict(), epoch = env.best.epoch, score = env.best.score)

        run_testing('test', env.dataset.test_images, model, env, hook=update('test'))

        save_checkpoint = struct(current = current, best = best, args = env.model_args, run = env.run)
        torch.save(save_checkpoint, env.model_path)

        send_command("checkpoint", ((env.run, env.epoch), score, is_best))
        env.epoch = env.epoch + 1

        if args.detections > 0 and conn:
            results = run_detections(model, env, hook=update('detect'), n=args.detections)
            send_command('detections', results)

        if env.pause_time is not None:
            env.pause_time = env.pause_time - 1

            if env.pause_time == 0:
                raise UserCommand('pause')

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
        validate = training_cycle,
        pause = paused
    )

    if conn is not None:
        activity = paused if args.paused else training_cycle

        while(True):
            try:
                activity()
                poll_command()

            except UserCommand as cmd:
                assert cmd.command in activities, "Unknown command " + cmd.command

                print ("User command: ", cmd.command)
                activity = activities[cmd.command]
    else:
        try:
            while(True):
                training_cycle()

        except UserCommand as cmd:                
            pass

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
