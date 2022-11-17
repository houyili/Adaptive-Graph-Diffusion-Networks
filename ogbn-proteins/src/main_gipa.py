#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import dgl.function as fn
from dgl.dataloading import MultiLayerFullNeighborSampler, MultiLayerNeighborSampler, DataLoader
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from new_sample import EdgeSampleNeighborSampler, InSeedNodeNeighborSampler
from gipa import GIPA
from utils import get_cpu_list

device = None
dataset = "ogbn-proteins"
n_node_feats, n_edge_feats, n_classes = 0, 8, 112


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def load_data(dataset, root):
    data = DglNodePropPredDataset(name=dataset, root=root)
    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    graph.ndata["labels"] = labels

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph, labels, train_idx):
    global n_node_feats

    # The sum of the weights of adjacent edges is used as node features.
    graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))
    n_node_feats = graph.ndata["feat"].shape[-1]

    # Only the labels in the training set are used as features, while others are filled with zeros.
    graph.ndata["train_labels_onehot"] = torch.zeros(graph.number_of_nodes(), n_classes)
    graph.ndata["train_labels_onehot"][train_idx, :] = labels[train_idx, :].float()
    graph.ndata["deg"] = graph.out_degrees().float().clamp(min=1)

    graph.create_formats_()

    return graph, labels


def gen_model(args):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats

    model = GIPA(
        n_node_feats_,
        n_edge_feats,
        n_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_hidden=args.n_hidden,
        edge_emb=16,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        attn_drop=args.attn_drop,
        edge_drop=args.edge_drop, first_layer_hidden=args.first_layer_hidden,
        use_attn_dst=not args.no_attn_dst,
        agg_batch_norm=args.agg_batch_norm, edge_att_act=args.edge_att_act
    )

    return model


def add_labels(graph, idx):
    feat = graph.srcdata["feat"]
    train_labels_onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    train_labels_onehot[idx] = graph.srcdata["train_labels_onehot"][idx]
    graph.srcdata["feat"] = torch.cat([feat, train_labels_onehot], dim=-1)


def train(args, model, dataloader, _labels, _train_idx, criterion, optimizer, _evaluator):
    model.train()

    loss_sum, total = 0, 0

    for input_nodes, output_nodes, subgraphs in dataloader:
        subgraphs = [b.to(device) for b in subgraphs]
        new_train_idx = torch.arange(len(output_nodes), device=device)

        if args.use_labels:
            train_labels_idx = torch.arange(len(output_nodes), len(input_nodes), device=device)
            train_pred_idx = new_train_idx

            add_labels(subgraphs[0], train_labels_idx)
        else:
            train_pred_idx = new_train_idx

        pred = model(subgraphs)
        loss = criterion(pred[train_pred_idx], subgraphs[-1].dstdata["labels"][train_pred_idx].float())
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        optimizer.step()

        count = len(train_pred_idx)
        loss_sum += loss.item() * count
        total += count

        torch.cuda.empty_cache()

    return loss_sum / total


@torch.no_grad()
def evaluate(args, model, dataloader, labels, train_idx, val_idx, test_idx, criterion, evaluator):
    model.eval()

    preds = torch.zeros(labels.shape).to(device)

    # Due to the memory capacity constraints, we use sampling for inference and calculate the average of the predictions 'eval_times' times.
    eval_times = 1

    for _ in range(eval_times):
        for input_nodes, output_nodes, subgraphs in dataloader:
            subgraphs = [b.to(device) for b in subgraphs]
            new_train_idx = list(range(len(input_nodes)))

            if args.use_labels:
                add_labels(subgraphs[0], new_train_idx)

            pred = model(subgraphs)
            preds[output_nodes] += pred

            torch.cuda.empty_cache()

    preds /= eval_times

    train_loss = criterion(preds[train_idx], labels[train_idx].float()).item()
    val_loss = criterion(preds[val_idx], labels[val_idx].float()).item()
    test_loss = criterion(preds[test_idx], labels[test_idx].float()).item()

    return (
        evaluator(preds[train_idx], labels[train_idx]),
        evaluator(preds[val_idx], labels[val_idx]),
        evaluator(preds[test_idx], labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
        preds,
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred, "y_true": labels})["rocauc"]

    # train_batch_size = (len(train_idx) + 9) // 10
    train_batch_size = (len(train_idx) + args.batch_rate - 1) // args.batch_rate
    eval_batch_size = (len(train_idx) + len(val_idx) + len(test_idx) + args.eval_batch_rate - 1) // args.eval_batch_rate
    # batch_size = len(train_idx)

    if args.sample_no_limit:
        sample_num = [-1] * args.n_layers
    else:
        sample_num = [args.sample1,args.sample2,args.sample3,args.sample4,args.sample5,args.sample6]

    if args.sample_type == "edge_rate_sample" and args.edge_sample_rate > 0 and args.edge_sample_rate < 1:
        train_sampler = EdgeSampleNeighborSampler(sample_num, args.edge_sample_rate, min_fanout=1)
        eval_sampler = EdgeSampleNeighborSampler(sample_num, 0.4, min_fanout=1)
    elif args.sample_type == "in_seed_sample":
        train_sampler = InSeedNodeNeighborSampler(sample_num)
        eval_sampler = InSeedNodeNeighborSampler([3 * i if i > 0 else -1 for i in sample_num])
    else:
        train_sampler = MultiLayerNeighborSampler(sample_num)
        eval_sampler = MultiLayerNeighborSampler([3 * i if i > 0 else -1 for i in sample_num])

    train_dataloader = DataLoader(graph.cpu(), train_idx.cpu(), train_sampler, batch_size=train_batch_size, num_workers=10)
    eval_dataloader =  DataLoader(graph.cpu(), torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]),
                                  eval_sampler,  batch_size=eval_batch_size, num_workers=10)


    criterion = nn.BCEWithLogitsLoss()

    model = gen_model(args).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('params')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=50, verbose=True)

    total_time = 0
    val_score, best_val_score, final_test_score = 0, 0, 0

    train_scores, val_scores, test_scores = [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []
    final_pred = None

    cpu1,cpu2 = get_cpu_list(args.cpu_start_from, 10)
    with train_dataloader.enable_cpu_affinity(loader_cores=cpu1, compute_cores=cpu2):
        with eval_dataloader.enable_cpu_affinity(loader_cores=cpu1, compute_cores=cpu2):
            for epoch in range(1, args.n_epochs + 1):
                tic = time.time()

                loss = train(args, model, train_dataloader, labels, train_idx, criterion, optimizer, evaluator_wrapper)

                toc = time.time()
                total_time += toc - tic

                if epoch == args.n_epochs or epoch % args.eval_every == 0 or epoch % args.log_every == 0:
                    train_score, val_score, test_score, train_loss, val_loss, test_loss, pred = evaluate(
                        args, model, eval_dataloader, labels, train_idx, val_idx, test_idx, criterion, evaluator_wrapper
                    )

                    if val_score > best_val_score:
                        best_val_score = val_score
                        final_test_score = test_score
                        final_pred = pred

                    if epoch % args.log_every == 0:
                        print(
                            f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}s"
                        )
                        print(
                            f"Loss: {loss:.4f}\n"
                            f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                            f"Train/Val/Test/Best val/Final test score: {train_score:.4f}/{val_score:.4f}/{test_score:.4f}/{best_val_score:.4f}/{final_test_score:.4f}"
                        )

                    for l, e in zip(
                        [train_scores, val_scores, test_scores, losses, train_losses, val_losses, test_losses],
                        [train_score, val_score, test_score, loss, train_loss, val_loss, test_loss],
                    ):
                        l.append(e)

                lr_scheduler.step(val_score)

    print("*" * 50)
    print(f"Best val score: {best_val_score}, Final test score: {final_test_score}")
    print("*" * 50)

    if args.save_pred:
        os.makedirs("./output", exist_ok=True)
        torch.save(F.softmax(final_pred, dim=1), f"./output/{n_running}.pt")

    return best_val_score, final_test_score


def count_parameters(args):
    model = gen_model(args)
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():
    global device

    argparser = argparse.ArgumentParser("GAT implementation on ogbn-proteins", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--batch_rate", type=int, default=10)
    argparser.add_argument("--eval-batch-rate", type=int, default=5)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides '--gpu'.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    argparser.add_argument("--seed", type=int, default=0, help="random seed")
    argparser.add_argument("--n-runs", type=int, default=10, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=1200, help="number of epochs")
    argparser.add_argument("--use-labels", action="store_true", help="Use labels in the training set as input features.")
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--n-heads", type=int, default=6, help="number of heads")
    argparser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=6, help="number of layers")
    argparser.add_argument("--n-hidden", type=int, default=80, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.1, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention dropout rate")
    argparser.add_argument("--edge-drop", type=float, default=0.1, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--eval-every", type=int, default=5, help="evaluate every EVAL_EVERY epochs")
    argparser.add_argument("--log-every", type=int, default=5, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot", action="store_true", help="plot learning curves")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")
    argparser.add_argument("--root", type=str, default="./datasets")
    argparser.add_argument("--cpu-start-from", type=int, default=0)
    argparser.add_argument("--agg-batch-norm", action="store_true", help="enable batch norm in agg")
    argparser.add_argument("--edge-att-act", type=str, default="leaky_relu", choices=["leaky_relu", "tanh", "softplus"])
    argparser.add_argument("--first-layer-hidden", type=int, default=150)
    argparser.add_argument("--edge-sample-rate", type=float, default=0.1)
    argparser.add_argument("--sample-no-limit", action="store_true")
    argparser.add_argument("--sample1", type=int, default=32)
    argparser.add_argument("--sample2", type=int, default=32)
    argparser.add_argument("--sample3", type=int, default=32)
    argparser.add_argument("--sample4", type=int, default=32)
    argparser.add_argument("--sample5", type=int, default=32)
    argparser.add_argument("--sample6", type=int, default=32)
    argparser.add_argument("--sample-type", type=str, default="neighbor_sample",
                           choices=["neighbor_sample", "in_seed_sample", "edge_rate_sample"])

    args = argparser.parse_args()
    print(args)
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # load data & preprocess
    print("Loading data")
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, root=args.root)
    print("Preprocessing")
    graph, labels = preprocess(graph, labels, train_idx)

    labels, train_idx, val_idx, test_idx = map(lambda x: x.to(device), (labels, train_idx, val_idx, test_idx))

    # run
    val_scores, test_scores = [], []

    for i in range(args.n_runs):
        print("Running", i)
        seed(args.seed + i)
        val_score, test_score = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i + 1)
        val_scores.append(val_score)
        test_scores.append(test_score)

    print(" ".join(sys.argv))
    print(args)
    print(f"Runned {args.n_runs} times")
    print("Val scores:", val_scores)
    print("Test scores:", test_scores)
    print(f"Average val score: {np.mean(val_scores)} ± {np.std(val_scores)}")
    print(f"Average test score: {np.mean(test_scores)} ± {np.std(test_scores)}")
    print(f"Number of params: {count_parameters(args)}")


if __name__ == "__main__":
    main()
