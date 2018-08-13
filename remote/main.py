import torch
import torch.optim as optim

import json

from json.decoder import JSONDecodeError

from remote.dataset import decode_dataset
from remote.connection import connect

from detection.models import models
from tools.model import io

from tools import Struct
from arguments import parameters
from tools.parameters import default_parameters

from detection.loss import total_bce

import time
import trainer
import evaluate


def make_command(name, contents):
    if contents is None:
        return {'tag':name}
    else:
        return {'tag':name, 'contents':contents}

def match_command(cmd):
    return cmd['tag'], cmd['contents'] if 'contents' in cmd else None


def ready(*xs):
    return all(v is not None for v in xs)


class Reset(Exception):
    def __init__(self, env):
        self.env = env






def initialise(dataset, params):
    model_args = Struct(num_classes = len(dataset.classes), input_channels = 3)

    model, encoder = io.create(models, Struct (model='fcn'), model_args)
    optimizer = optim.SGD(model.parameter_groups(params.lr, params.fine_tuning), lr=params.lr, momentum=params.momentum)

    params = params

    best = 0.0
    epoch = 0

    output_path = "output"

    return Struct(**locals())


def train(conn):

    params = default_parameters(parameters).merge(Struct(
        batch_size = 4,
        num_workers = 4,
        image_size = 440
    ))

    print(params)

    env = None
    device = torch.cuda.current_device()

    def process_command(str):
        try:
            tag, data = match_command(json.loads(str))
            print("recieved command: " + tag)

            if tag == 'TrainerDataset':
                env = initialise(decode_dataset(data), params)

                raise Reset(env)

            elif tag == 'TrainerUpdate':
                update = decode_update(data)

            elif tag == 'TrainerDetect':
                detect = decode_detect(data)
            else:
                assert False, "unknown command: " + tag

        except (JSONDecodeError) as err:
            conn.send(make_command('TrainerError', repr(err)))
            return None


    def poll_command():
        if conn.poll():
            cmd = conn.recv()
            process_command(cmd)


    def training_cycle():
        env.model.to(device)

        stats = trainer.train(env.model, env.dataset.sample_train(params, env.encoder),
                    evaluate.eval_train(total_bce, device), env.optimizer, check=poll_command)

        summarize_train("train", stats, env.epoch)

        stats = trainer.test(env.model, env.dataset.test(args), evaluate.eval_test(env.encoder, device), check=poll_command)
        score = summarize_test("test", stats, env.epoch)

        if score >= env.best:
            io.save(env.output_path, env.model, env.model_args, env.epoch, score)
            env.best = score

        env.epoch = env.epoch + 1


    while(True):
        try:
            if env is not None:
                training_cycle()
            poll_command()

        except Reset as reset:
            env   = reset.env

def run_main():
    p, conn = connect('ws://localhost:2160')

    try:
        train(conn)
    except (KeyboardInterrupt, SystemExit):
        p.terminate()




if __name__ == '__main__':
    run_main()
