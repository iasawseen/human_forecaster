import os
import sys

import torch
import importlib
import shutil
import logging
import pickle
import random
import numpy as np
import json


def save_json(obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(json.dumps(obj), f)


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.loads(json.load(f))


def save(obj, file_path):
    print('saving to {}'.format(file_path))
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

