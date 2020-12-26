#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import time
import random

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import dgl
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from models import SGATHA
from logger import get_logger
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = None
in_feats, n_classes = None, None
epsilon = 1 - math.log(2)


def gen_model(args):
    norm = 'both' if args.use_norm else None
    if args.use_labels:
        model = SGATHA(
            in_feats + n_classes,
            n_classes,
            K=args.K,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            norm=norm
        )
    else:
        model = SGATHA(
            in_feats,
            n_classes,
            K=args.K,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            norm=norm
        )

    return model

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]


def add_labels(feat, labels, idx):
    onehot = th.zeros([feat.shape[0], n_classes]).to(device)
    onehot[idx, labels[idx, 0]] = 1
    return th.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(model, graph, labels, train_idx, optimizer, use_labels):
    model.train()

    feat = graph.ndata["feat"]

    if use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
    loss.backward()
    optimizer.step()

    return loss, pred


@th.no_grad()
def evaluate(model, graph, labels, train_idx, val_idx, test_idx, use_labels, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)
    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running, logger):
    # define model and optimizer
    model = gen_model(args)
    model = model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training loop
    total_time = 0
    best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        # adjust_learning_rate(optimizer, args.lr, epoch)

        loss, pred = train(model, graph, labels, train_idx, optimizer, args.use_labels)
        acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss = evaluate(
            model, graph, labels, train_idx, val_idx, test_idx, args.use_labels, evaluator
        )

        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % args.log_every == 0:
            logger.info(f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}")
            logger.info(f"Time: {(total_time / epoch):.4f}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")
            logger.info(f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}")
            logger.info(f"Train/Val/Test/Best val/Best test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{best_test_acc:.4f}")

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss.item(), train_loss, val_loss, test_loss],
        ):
            l.append(e)

    logger.info("*" * 50)
    logger.info(f"Average epoch time: {total_time / args.n_epochs}, Test acc: {best_test_acc}")

    if args.plot_curves:
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.set_yticks(np.linspace(0, 1.0, 101))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip([accs, train_accs, val_accs, test_accs], ["acc", "train acc", "val acc", "test acc"]):
            plt.plot(range(args.n_epochs), y, label=label)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_acc_{n_running}.png")

        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [losses, train_losses, val_losses, test_losses], ["loss", "train loss", "val loss", "test loss"]
        ):
            plt.plot(range(args.n_epochs), y, label=label)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_loss_{n_running}.png")

    return best_val_acc, best_test_acc


def count_parameters(args):
    model = gen_model(args)
    print([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():
    global device, in_feats, n_classes, epsilon

    argparser = argparse.ArgumentParser("HGAT on OGBN-Arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--seed", type=int, default=0, help="initial random seed.")
    argparser.add_argument("--n-runs", type=int, default=10)
    argparser.add_argument("--n-epochs", type=int, default=2000)
    argparser.add_argument(
        "--use-labels", action="store_true", help="Use labels in the training set as input features."
    )
    argparser.add_argument("--use-norm", action="store_true", help="Use symmetrically normalized adjacency matrix.")
    argparser.add_argument("--lr", type=float, default=0.002)
    argparser.add_argument("--n-layers", type=int, default=3)
    argparser.add_argument("--K", type=int, default=9)
    argparser.add_argument("--n-heads", type=int, default=3)
    argparser.add_argument("--n-hidden", type=int, default=256)
    argparser.add_argument("--dropout", type=float, default=0.75)
    argparser.add_argument("--input_drop", type=float, default=0.25)
    argparser.add_argument("--attn_drop", type=float, default=0.05)
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--plot-curves", action="store_true")
    argparser.add_argument("--log-path", type=str, default="../logs/gat-ha/")
    args = argparser.parse_args()

    if args.cpu:
        device = th.device("cpu")
    else:
        device = th.device("cuda:%d" % args.gpu)

    # load data
    data = DglNodePropPredDataset(name="ogbn-arxiv", root="../dataset")
    evaluator = Evaluator(name="ogbn-arxiv")

    args_dict = dict(args._get_kwargs())
    exclude_names = ['cpu' ,'gpu', 'log_every', 'log_path', 'plot_curves']
    logger_args = []
    for u,v in args_dict.items():
        if u not in exclude_names:
            logger_args.extend([u, str(v)])
    logger = get_logger(os.path.join(args.log_path, "_".join(logger_args) + ".log"))

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    logger.debug(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    logger.debug(f"Total edges after adding self-loop {graph.number_of_edges()}")

    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    # graph.create_format_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)

    # run
    val_accs = []
    test_accs = []

    for i in range(1, args.n_runs + 1):
        seed(i + args.seed)
        val_acc, test_acc = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i, logger)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    logger.info(f"Runned {args.n_runs} times")
    logger.info(f"Val Accs: {val_accs}")
    logger.info(f"Test Accs: {test_accs}")
    logger.info(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    logger.info(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    logger.info(f"Number of params: {count_parameters(args)}")
    logger.info(f"args: {args}")

if __name__ == "__main__":
    main()



