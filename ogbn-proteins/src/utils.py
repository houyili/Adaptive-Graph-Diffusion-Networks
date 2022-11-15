import random
from time import sleep

import dgl

import numpy as np
import torch
import math
import torch.nn.functional as F


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def compute_norm(graph):
    degs = graph.in_degrees().float().clamp(min=1)
    deg_isqrt = torch.pow(degs, -0.5)

    degs = graph.in_degrees().float().clamp(min=1)
    deg_sqrt = torch.pow(degs, 0.5)

    return deg_sqrt, deg_isqrt

def add_labels(graph, idx, n_classes, device):
    feat = graph.srcdata["feat"]
    train_labels_onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    train_labels_onehot[idx] = graph.srcdata["train_labels_onehot"][idx]
    graph.srcdata["feat"] = torch.cat([feat, train_labels_onehot], dim=-1)


def loge_BCE(x, labels):
    epsilon = 1 - math.log(2)
    y = F.binary_cross_entropy_with_logits(x, labels, reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def print_msg_and_write(out_msg, log_f):
    print(out_msg)
    log_f.write(out_msg)
    log_f.flush()

