
import json
from json.decoder import JSONDecodeError

from remote.dataset import decode_dataset
from remote.connection import connect

import time


def make_command(name, contents):
    if contents is None:
        return {'tag':name}
    else:
        return {'tag':name, 'contents':contents}

def match_command(cmd):
    return cmd['tag'], cmd['contents'] if 'contents' in cmd else None


def train(conn):

    def process_command(str):

        try:
            tag, data = match_command(json.loads(str))

            if tag == 'TrainerDataset':
                dataset = decode_dataset(data)

            elif tag == 'TrainerUpdate':
                update = decode_update(data)

            elif tag == 'TrainerDetect':
                detect = decode_detect(data)
            else:
                assert False, "unknown command: " + tag

        except (JSONDecodeError) as err:
            conn.send(make_command('TrainerError', repr(err)))
            return None


    n = 0
    while(True):

        if conn.poll():
            cmd = conn.recv()

            print ("got", cmd)
            process_command(cmd)



        print ('.')
        time.sleep(0.25)


def run_main():
    p, conn = connect('ws://localhost:2160')

    try:
        train(conn)
    except (KeyboardInterrupt, SystemExit):
        p.terminate()




if __name__ == '__main__':
    run_main()
