import random
import torch
import dgl

import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
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

def get_cpu_list(start_from:int, cpu_num:int):
    cpu_num_2 = int(float(cpu_num) * 1.5)
    base = start_from
    list_1 = [i + base for i in range(cpu_num)]
    list_2 = [i + base + cpu_num for i in range(cpu_num_2)]
    return list_1,list_2

def get_act_by_str(name:str, negative_slope:float=0):
    if name == "leaky_relu":
        res = nn.LeakyReLU(negative_slope, inplace=True)
    elif name == "tanh":
        res = nn.Tanh()
    elif name == "none":
        res = nn.Identity()
    elif name == "relu":
        res = nn.ReLU()
    else:
        res = nn.Softplus()
    return res


class FocalLoss(nn.modules.loss.BCEWithLogitsLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss