import torch

import time
import os
import json
import copy
import random

import sys
import traceback

from torch import nn
import torch.optim as optim

from json.decoder import JSONDecodeError

from dataset.annotate import decode_dataset, split_tagged, tagged, decode_image, init_dataset, decode_object_map
from dataset.imports import load_dataset

from dataset.detection import least_recently_evaluated

from detection import models, box, detection_table, export

import checkpoint

import tools

from tools.parameters import default_parameters, get_choice
from tools import table, struct, logger, to_structs, Struct, window, tensors_to, shape

from tools.logger import EpochLogger

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



def initialise(config, dataset, args):

    data_root = config.root
    log_root = args.log_dir or data_root

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

    output_path, log = logger.make_experiment(log_root, args.run_name, load=not args.no_load, dry_run=args.dry_run)
    model_path = os.path.join(output_path, "model.pth")

    model, encoder = models.create(model_args.model, model_args.dataset)

    set_bn_momentum(model, args.bn_momentum)

    best, current, resumed = checkpoint.load_checkpoint(model_path, model, model_args, args)
    model, epoch = current.model, current.epoch + 1

    pause_time = args.pause_epochs
    running_average = [] if epoch >= args.average_start else []


    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    device = torch.cuda.current_device()
    tests = args.tests.split(",")

    return struct(**locals())


def make_detections(env, predictions):
    return export.make_detections(predictions, 
        classes=env.dataset.classes, thresholds=env.best.thresholds, 
        scale=env.scale, network_id = (env.run, env.epoch))


def evaluate_detections(env, image, nms_params):
    model = env.best.model
    detections = evaluate.evaluate_image(model.to(env.device), image, env.encoder.to(env.device), 
        nms_params=nms_params, device=env.device).detections
    return make_detections(list(detections._sequence()))


def table_list(t):
    return list(t._sequence())

def evaluate_review(env, image, nms_params, review):
    model = env.best.model.to(env.device)
    encoder = env.encoder.to(env.device)

    scale = env.args.scale

    detections = evaluate.evaluate_image(model, image, encoder, 
        device=env.device, nms_params=nms_params).detections

    if detections._size == 0:
        return make_detections(env, [])

    review = tensors_to(review, device=env.device)
    ious = box.iou_matrix(detections.bbox, review.bbox * scale)
    
    ious[ious < nms_params.nms].fill_(-1)
    scores = ious.mul(detections.confidence.unsqueeze(1))

    review_inds = scores.max(0).values.argsort(descending=True)

    detections = table_list(detections)
    for i in review_inds.tolist():
        score, ind = scores[: , i].max(0)
        
        if score > 0:
            detections[ind].match = i

        scores[ind].fill_(0)

    return make_detections(env, detections)



def detect_request(env, file, nms_params, review=None):
    path = os.path.join(env.data_root, file)

    if not os.path.isfile(path):
        raise NotFound(file)

    image = env.dataset.load_inference(file, path, env.args)

    if review is None:
        return evaluate_detections(env, image, nms_params)
    else:
        return evaluate_review(env, image, nms_params, review)

def log_anneal(range, t):
    begin, end = range
    return math.exp(math.log(begin) * (1 - t) + math.log(end) * t)

def cosine_anneal(range, t):
    begin, end = range
    return end + 0.5 * (begin - end) * (1 + math.cos(t * math.pi))

def schedule_lr(t, epoch, args):
    lr_min = args.lr * args.lr_min

    if args.lr_decay == "log":
        return log_anneal((args.lr, lr_min), t) 
    elif args.lr_decay == "cosine":
        return cosine_anneal((args.lr, lr_min), t) 
    elif args.lr_decay == "step":
        n = math.floor(epoch / args.lr_schedule)
        return max(lr_min, args.lr * math.pow(args.lr_step, -n))
    else:
        assert False, "unknown lr decay method: " + args.lr_decay


def adjust_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        modified = lr * param_group['modifier'] if 'modifier' in param_group else lr
        param_group['lr'] = modified




def set_bn_momentum(model, mom):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = mom

def get_nms_params(args):
    return struct(
            nms = args.nms_threshold,
            threshold = args.class_threshold,
            detections = args.max_detections)

def test_images(images, model, env, split=False, hook=None):
    eval_params = struct(
        overlap = env.args.overlap,
        split = split,
        image_size = (env.args.train_size, env.args.train_size),
        batch_size = env.args.batch_size,
        nms_params = get_nms_params(env.args),
        device = env.device,
        debug = env.debug
    )

    eval_test = evaluate.eval_test(model.eval(), env.encoder, eval_params)
    return trainer.test(env.dataset.test_on(images, env.args, env.encoder), eval_test, hook=hook)


def run_testing(name, images, model, env, split=False, hook=None, thresholds=None):

  if len(images) > 0:
      print("{} {}:".format(name, env.epoch))
      results = test_images(images, model, env, split=split, hook=hook)

      return evaluate.summarize_test(name, results, env.dataset.classes, env.epoch, 
        log=EpochLogger(env.log, env.epoch), thresholds=thresholds)

  return 0, None


def class_counts(detection, class_id):
    t, count = detection.stats.class_counts[class_id].middle
    return count


def run_detections(model, env, images, hook=None, variation_window=None):
    if len(images) > 0:
        images = sorted(images, key = lambda img: img.key)
        results = test_images(images, model, env, hook=hook)

        mask = torch.ByteTensor([image.category in ['discard'] for image in images])
    
        detections = [make_detections(env, table_list(result.detections)) for result in results]

        if variation_window is not None:
            variation = torch.Tensor(len(images)).zero_()
            for c in env.dataset.classes:
                counts = [class_counts(detection, c.id) for detection in detections]
                counts = torch.Tensor(counts)

                variation += window.masked_diff(counts, mask=mask, window=variation_window)

            for v, d in zip(variation, detections):
                d.stats.frame_variation = v

        return {image.id : d for image, d in zip(images, detections)}
    


def report_training(results):
    images = {}

    # for r in results:
    #     for file, loss in r.files:
    #         add_multimap(images, file, struct(loss = loss))

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

                args.no_load = False # For subsequent initialisations

                if not args.paused:
                    raise UserCommand('resume')
            
            elif tag == 'import':
                file, image_data = data

                image = decode_image(image_data, env.config)
                env.dataset.update_image(image)

            elif tag == 'update':
                file, method, image_data = data

                image = decode_image(image_data, env.config)
                env.dataset.update_image(image)

                if image.category == 'validate':
                    env.best.score = 0

                if env.pause_time == 0:
                    env.pause_time = env.args.pause_epochs
                    raise UserCommand('resume')
                else:
                    env.pause_time = env.args.pause_epochs

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
                print ("unknown command: " + tag)


        except (JSONDecodeError) as err:
            send_command('error', repr(err))
            return None

    def poll_command():
        while conn and conn.poll():
            cmd = conn.recv()
            process_command(cmd)

    def train_update(n, total):

        lr = schedule_lr(n/total, env.epoch, args)
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

        if args.max_epochs is not None and env.epoch > args.max_epochs:
            raise UserCommand('pause')


        log = EpochLogger(env.log, env.epoch)
        model = env.model.to(env.device)
        encoder = env.encoder.to(env.device)

        log.scalars("dataset", Struct(env.dataset.count_categories()))

        train_images = env.dataset.train_images
        if args.incremental is True:
            t = env.epoch / args.max_epochs
            n = max(1, min(int(t * len(train_images)), len(train_images)))
            train_images = train_images[:n]

        print("training {} on {} images:".format(env.epoch, len(train_images)))
        train_stats = trainer.train(env.dataset.sample_train_on(train_images, args, env.encoder),
            evaluate.eval_train(model.train(), env.encoder, env.debug, 
            device=env.device), env.optimizer, hook=train_update)

        evaluate.summarize_train("train", train_stats, env.dataset.classes, env.epoch, log=log)


        send_command('training', report_training(train_stats))


        score, thresholds = run_testing('validate', env.dataset.validate_images, model, env,  hook=update('validate'))
        if env.args.eval_split:           
            run_testing('validate_split', env.dataset.validate_images, model, env, split=True, hook=update('validate'))            


        is_best = score >= env.best.score
        if is_best:
            env.best = struct(model = copy.deepcopy(model), score = score, thresholds = thresholds, epoch = env.epoch)


        current = struct(state = model.state_dict(), epoch = env.epoch, thresholds = thresholds, score = score)
        best = struct(state = env.best.model.state_dict(), epoch = env.best.epoch, thresholds = env.best.thresholds, score = env.best.score)

        # run_testing('test', env.dataset.test_images, model, env, hook=update('test'))
        
        for test_name in env.tests:
            run_testing(test_name, env.dataset.get_images(test_name), model, env, 
                hook=update('test'), thresholds = env.best.thresholds)                
        
        save_checkpoint = struct(current = current, best = best, args = env.model_args, run = env.run)
        torch.save(save_checkpoint, env.model_path)

        send_command("checkpoint", ((env.run, env.epoch), score, is_best))
        env.epoch = env.epoch + 1

        if (args.detections > 0) and conn:
            detect_images = least_recently_evaluated(env.dataset.new_images, n = args.detections)

            results = run_detections(model, env, detect_images, hook=update('detect'))
            send_command('detections', results)

        if args.detect_all and conn:
            detect_images = env.dataset.get_images()

            results = run_detections(model, env, detect_images, hook=update('detect'), variation_window=args.variation_window)
            send_command('detections', results)


        # if env.best.epoch < env.epoch - args.validation_pause:
        #     raise UserCommand('pause')

        if env.pause_time is not None:
            env.pause_time = env.pause_time - 1

            if env.pause_time == 0:
                raise UserCommand('pause')

        log.flush()


    def review_all():
        print("reviewing...")

    def detect_all():
        print("detecting...")
        model = env.model.to(env.device)
        encoder = env.encoder.to(env.device)

        detect_images = env.dataset.get_images()

        results = run_detections(model, env, detect_images, hook=update('detect'), variation_window=args.variation_window)
        send_command('detections', results)

        raise UserCommand('resume')

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
    # torch.multiprocessing.set_start_method('spawn')
    run_main()
