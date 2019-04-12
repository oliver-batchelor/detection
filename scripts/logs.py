from os import path
from tools import struct, to_structs, concat_lists, to_dicts, pluck, pprint_struct, transpose_structs, Struct, append_dict, transpose_dicts

import math
import json






def read_log(file):
    entries = [to_structs(json.loads(line)) for line in open(file, mode="r")]

    steps = {}
    tags = {}
    for entry in entries:
        step = steps.get(entry.step) or {}
        step[entry.tag] = entry.value

        tags[entry.tag] = True
        steps[entry.step] = step

    return struct (tags=tags.keys(), steps={i : Struct(entry) for i, entry in steps.items()})





def sortdict(d, **opts):
    # **opts so any currently supported sorted() options can be passed
    for k in sorted(d, **opts):
        yield k, d[k]

def get_entry(log, name):
    return {i:entry[name] for i, entry in log.steps.items() if name in entry}


def get_keys(log, prefix):
    keys = [tag[len(prefix) + 1:] for tag in log.tags if tag.startswith(prefix)]
    entries = {k:get_entry(log, prefix + "/" + k) for k in keys}
    return transpose_dicts(entries)


def get_prs(log, category='validate'):
    return get_keys(log, category + "/pr")
    


log_files = struct(
    #seals = 'seals.json'
    scott_base = 'scott_base',
    fisheye = 'victor',

)    


def read_logs(base_path, log_files):

    def load(filename):
        filename = path.join(base_path, run_name, "log.json")
        if path.isfile(filename):
            return read_log(filename)

    return log_files._map_maybe(load)


if __name__ == '__main__':


    read_logs('/home/oliver/logs/validate', log_files)

    # d = get_entry(logs.scott_base, 'dataset')
    # print(d)


    #print( pluck('mAP50', logs.scott_base['validate']) )
