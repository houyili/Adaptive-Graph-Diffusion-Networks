#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from data import load_data, preprocess
from gen_model import count_parameters, gen_model, MODEL_LIST, count_model_parameters
from sampler import  random_partition_v2
from utils import add_labels, seed, print_msg_and_write

os.chdir(os.path.dirname(os.path.abspath(__file__)))
device = None
dataset = "ogbn-proteins"
n_node_feats, n_edge_feats, n_classes = 0, 8, 112


def train(args, graph, model, _labels, _train_idx, criterion, optimizer, _evaluator):
    model.train()
    train_pred = torch.zeros(graph.ndata["labels"].shape).to(device)
    loss_sum, total = 0, 0

    for batch_nodes, subgraph in random_partition_v2(args.train_partition_num, graph, shuffle=True):
        nodes_id = torch.from_numpy(batch_nodes).to(device)
        subgraph = subgraph.to(device)
        pred = model(subgraph)
        train_pred[nodes_id] += pred
        train_pred_idx = nodes_id[torch.isin(nodes_id, _train_idx)].to(device)
        loss = criterion(pred[train_pred_idx], subgraph.ndata["labels"][train_pred_idx].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(batch_nodes)
        total += len(batch_nodes)
    train_score = _evaluator(train_pred[_train_idx], _labels[_train_idx])

    return loss_sum / total, train_score


@torch.no_grad()
def evaluate(args, graph, model, labels, train_idx, val_idx, test_idx, criterion, evaluator):
    torch.cuda.empty_cache()
    model.eval()

    preds = torch.zeros(labels.shape).to(device)
    for _ in range(args.eval_times):
        for batch_nodes, subgraph in random_partition_v2(args.eval_partition_num, graph, shuffle=False):
            subgraph = subgraph.to(device)
            pred = model(subgraph)
            preds[batch_nodes] += pred
    if args.eval_times > 1:
        preds /= args.eval_times

    train_loss = criterion(preds[train_idx], labels[train_idx].float()).item()
    val_loss = criterion(preds[val_idx], labels[val_idx].float()).item()
    test_loss = criterion(preds[test_idx], labels[test_idx].float()).item()
    train_score = evaluator(preds[train_idx], labels[train_idx])
    val_score = evaluator(preds[val_idx], labels[val_idx])
    test_score = evaluator(preds[test_idx], labels[test_idx])

    return (train_score, val_score, test_score, train_loss, val_loss, test_loss,[preds[train_idx], preds[val_idx], preds[test_idx]])


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running, log_f):
    evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred, "y_true": labels})["rocauc"]


    criterion = nn.BCEWithLogitsLoss()
    model = gen_model(args, n_node_feats, n_edge_feats, n_classes).to(device)

    lr_scheduler = None
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.advanced_optimizer:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=50, verbose=True)

    total_time, eval_time, eval_num = 0, 0, 0
    val_score, best_val_score, final_test_score, best_step = 0, 0, 0, 0
    train_scores, val_scores, test_scores = [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []
    final_pred = None

    for epoch in range(1, args.n_epochs + 1):
        if epoch == 1:
            title_msg = f"Number of node feature: {n_node_feats}\n" + f"Number of edge feature: {n_edge_feats}\n" + \
                        f"Number of params: {count_model_parameters(model)}\n"
            print_msg_and_write(title_msg, log_f)
        tic = time.time()

        loss, t_score = train(args, graph, model, labels, train_idx, criterion, optimizer, evaluator_wrapper)

        toc = time.time()
        total_time += toc - tic
        train_msg = f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, " \
                    f"this epoch time: {toc - tic:.2f}s Train loss/score: {loss:.4f}/{t_score:.4f}\n"
        print_msg_and_write(train_msg, log_f)

        if epoch == args.n_epochs or epoch % args.eval_every== 0 or epoch % args.log_every == 0:
            tic = time.time()
            train_score, val_score, test_score, train_loss, val_loss, test_loss, pred = evaluate(
                args, graph, model, labels, train_idx, val_idx, test_idx, criterion, evaluator_wrapper)
            eval_num += 1
            toc = time.time()
            eval_time += (toc - tic)

            if val_score > best_val_score:
                best_val_score = val_score
                final_test_score = test_score
                final_pred = pred
                best_step = epoch

            if epoch % args.log_every == 0:
                out_msg = f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, " \
                          f"Average Train epoch time: {total_time / epoch:.2f}s, " \
                          f"Average Eval epoch time: {eval_time / eval_num:.2f}s\n" \
                          f"Loss: {loss:.4f} Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n" \
                          f"Train/Val/Test: {train_score:.4f}/{val_score:.4f}/{test_score:.4f}\n" \
                          f"Best val/Final test score/Best Step: {best_val_score:.4f}/{final_test_score:.4f}/{best_step}\n"
                print_msg_and_write(out_msg, log_f)
        if args.advanced_optimizer:
            lr_scheduler.step(val_score)

    out_msg = "*" * 50 + f"\nBest val score: {best_val_score}, Final test score: {final_test_score}\n" + "*" * 50 + "\n"
    print_msg_and_write(out_msg, log_f)

    if args.plot:
        from plot_unit import plot_stats
        plot_stats(args, train_scores, val_scores, test_scores, losses, train_losses, val_losses, test_losses, n_running)

    if args.save_pred:
        os.makedirs("../output", exist_ok=True)
        torch.save(F.sigmoid(final_pred), f"../output/{n_running}.pt")

    return best_val_score, final_test_score


def main():
    global device, n_node_feats, n_edge_feats, n_classes, global_labels_idx, global_pred_idx

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--root", type=str, default="/mnt/ssd/ssd/dataset")
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides '--gpu'.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    argparser.add_argument("--seed", type=int, default=0, help="random seed")
    argparser.add_argument("--n-runs", type=int, default=10, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=1200, help="number of epochs")
    argparser.add_argument("--eval-times", type=int, default=1)
    argparser.add_argument("--advanced-optimizer", action="store_true")
    argparser.add_argument("--model", type=str, default="agdn", choices=MODEL_LIST)
    argparser.add_argument("--use-one-hot-feature", action="store_true")
    argparser.add_argument("--sample-type", type=str, default="random_cluster", 
        choices=["neighbor_sample", "random_cluster", "khop_sample", "m_cluster"])
    argparser.add_argument("--train-partition-num", type=int, default=10, 
        help="number of partitions for training, which only takes effect when sample_type==random_cluster")
    argparser.add_argument("--eval-partition-num", type=int, default=3, 
        help="number of partitions for evaluating, which only takes effect when sample_type==random_cluster")
    argparser.add_argument("--use-labels", action="store_true", 
        help="Use labels in the training set as input features.")
    argparser.add_argument("--mask-rate", type=float, default=0.5, 
        help="rate of labeled nodes at each epoch, which only takes effect when sample_type==random_cluster & use_labels=True")
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--n-heads", type=int, default=3, help="number of heads")
    argparser.add_argument("--norm", type=str, default="none", choices=["none", "adj", "avg"])
    argparser.add_argument("--disable-fea-trans-norm", action="store_true", help="disable batch norm in fea trans part")
    argparser.add_argument("--edge-att-act", type=str, default="leaky_relu", choices=["leaky_relu", "tanh", "softplus", "none", "relu"])
    argparser.add_argument("--edge-agg-mode", type=str, default="both_softmax", choices=["both_softmax", "single_softmax", "none_softmax"])
    argparser.add_argument("--weight-style", type=str, default="HA", choices=["sum", "mean", "HC", "HA"])
    argparser.add_argument("--K", type=int, default=3)
    argparser.add_argument("--sampler-K", type=int, default=6)
    argparser.add_argument("--sampler-budget", type=int, default=30)
    argparser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=6, help="number of layers")
    argparser.add_argument("--n-hidden", type=int, default=80, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.1, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention dropout rate")
    argparser.add_argument("--hop-attn-drop", type=float, default=0.0, help="hop-wise attention dropout rate")
    argparser.add_argument("--edge-drop", type=float, default=0.1, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--eval-every", type=int, default=5, help="evaluate every EVAL_EVERY epochs")
    argparser.add_argument("--log-every", type=int, default=5, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot", action="store_true", help="plot learning curves")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")
    argparser.add_argument("--log-file-name", type=str, default="")
    argparser.add_argument("--first-hidden", type=int, default=150, help="first layer size")
    argparser.add_argument("--disable-att-edge", action="store_true")
    argparser.add_argument("--use-prop-edge", action="store_true")
    argparser.add_argument("--edge-emb-size", type=int, default=16)
    argparser.add_argument("--n-hidden-per-head", type=int, default=30)
    argparser.add_argument("--edge-prop-size", type=int, default=0)
    args = argparser.parse_args()
    print(args)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # load data & preprocess
    print("Loading data")
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, args.root)
    print("Preprocessing")
    graph, labels = preprocess(graph, labels, train_idx, n_classes, one_hot_feat=args.use_one_hot_feature,
                               user_label=args.use_labels, user_adj=args.norm=="adj", user_avg=args.norm=="avg",
                               val_idx=val_idx, test_idx=test_idx)
    if args.use_one_hot_feature:
        n_node_feats = graph.ndata["feat"].shape[-1] + graph.ndata["x"].shape[-1]
    else:
        n_node_feats = graph.ndata["feat"].shape[-1]


    labels, train_idx, val_idx, test_idx = map(lambda x: x.to(device), (labels, train_idx, val_idx, test_idx))

    # run
    val_scores, test_scores = [], []
    version = str(int(time.time())) if args.log_file_name=="" else "%s_%d" %(args.log_file_name, int(time.time()))
    os.makedirs("%s/log" % (args.root), exist_ok=True)
    for i in range(args.n_runs):
        log_f = open("%s/log/%s_part%d.log" % (args.root, version, i) , mode='a')
        print_msg_and_write(args.__str__() + "\n", log_f)
        print("Running", i)
        seed(args.seed + i)
        val_score, test_score = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i + 1, log_f)
        val_scores.append(val_score)
        test_scores.append(test_score)
        log_f.close()

    print(" ".join(sys.argv))
    print(args)
    print(f"Runned {args.n_runs} times")
    print("Val scores:", val_scores)
    print("Test scores:", test_scores)
    print(f"Average val score: {np.mean(val_scores)} ± {np.std(val_scores)}")
    print(f"Average test score: {np.mean(test_scores)} ± {np.std(test_scores)}")
    print(f"Number of params: {count_parameters(args, n_node_feats, n_edge_feats, n_classes)}")


if __name__ == "__main__":
    main()


