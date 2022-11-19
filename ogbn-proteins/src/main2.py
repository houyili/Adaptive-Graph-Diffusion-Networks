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
import dgl
from dgl.dataloading import  MultiLayerNeighborSampler
from dgl.dataloading import NodeDataLoader
from torch import nn

from data import load_data, preprocess
from gen_model import count_parameters, gen_model, MODEL_LIST
from sampler import BatchSampler, DataLoaderWrapper, RandomPartitionSampler, ShaDowKHopSampler, random_partition_v2
from utils import add_labels, seed, loge_BCE, print_msg_and_write

os.chdir(os.path.dirname(os.path.abspath(__file__)))
device = None
dataset = "ogbn-proteins"
n_node_feats, n_edge_feats, n_classes = 0, 8, 112


def train(args, graph, model, dataloader, _labels, _train_idx, val_idx, test_idx, criterion, optimizer, _evaluator):
    model.train()
    train_score = -1
    train_pred = torch.zeros(graph.ndata["labels"].shape).to(device)
    loss_sum, total = 0, 0
    if args.sample_type == "neighbor_sample":
        for input_nodes, output_nodes, subgraphs in dataloader:
            for k in range(len(subgraphs)):
                subgraphs[k].dstdata["l"] = subgraphs[k].dstdata["l_global"]
            subgraphs = [b.to(device) for b in subgraphs]
            new_train_idx = torch.arange(len(output_nodes), device=device)

            if args.use_labels:
                train_labels_idx = torch.arange(len(output_nodes), len(input_nodes), device=device)
                train_pred_idx = new_train_idx

                add_labels(subgraphs[0], train_labels_idx, n_classes, device)
            else:
                train_pred_idx = new_train_idx
            
            pred = model(subgraphs)
            loss = criterion(pred[train_pred_idx], subgraphs[-1].ndata["labels"]["_N"][train_pred_idx].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = len(train_pred_idx)
            loss_sum += loss.item() * count
            total += count

    if args.sample_type == "random_cluster":
        global_labels_idx, global_pred_idx = None, None
        if args.use_labels:
            global_train_idx = _train_idx.cpu().clone()
            np.random.shuffle(global_train_idx[:50125])
            global_labels_idx = global_train_idx[:int(50125*args.mask_rate)]
            global_pred_idx = global_train_idx[int(50125*args.mask_rate):]
        for batch_nodes, subgraph in random_partition_v2(args.train_partition_num, graph, shuffle=True):
            subgraph = subgraph.to(device)
            new_train_idx = torch.tensor(np.random.permutation(len(batch_nodes)), device=device)

            if args.use_labels:
                train_labels_idx = new_train_idx[np.isin(batch_nodes[new_train_idx.cpu()], global_labels_idx)]
                train_pred_idx = new_train_idx[np.isin(batch_nodes[new_train_idx.cpu()], global_pred_idx)]
                # train_labels_idx = new_train_idx[:int(len(new_train_idx)*args.mask_rate)]
                # train_pred_idx = new_train_idx[int(len(new_train_idx)*args.mask_rate):]
                # train_pred_idx = train_labels_idx = new_train_idx

                add_labels(subgraph, train_labels_idx, n_classes, device)
            else:
                train_pred_idx = new_train_idx
            # fine_nodes_mask = ((subgraph.ndata["sub_deg"] > 1) | (subgraph.ndata["sub_deg"] / subgraph.ndata["deg"] > 0.01)).cpu().numpy()
            inner_train_mask = np.isin(batch_nodes[train_pred_idx.cpu()], _train_idx.cpu())
            train_pred_idx = train_pred_idx[inner_train_mask]
            pred = model(subgraph)
            train_pred[batch_nodes] += pred
            # if args.n_label_iters > 0:
            #     unlabel_idx = np.setdiff1d(new_train_idx, train_labels_idx)
            #     for _ in range(args.n_label_iters):
            #         pred = pred.detach()
            #         torch.cuda.empty_cache()
            #         subgraph.ndata['feat'][unlabel_idx, -2*n_classes:-n_classes] = torch.sigmoid(pred[unlabel_idx])
            #         subgraph.ndata['feat'][unlabel_idx, -n_classes:] = 1-torch.sigmoid(pred[unlabel_idx])
            #         pred = model(subgraph)
            loss = criterion(pred[train_pred_idx], subgraph.ndata["labels"][train_pred_idx].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = len(train_pred_idx)
            loss_sum += loss.item() * count
            total += count
        train_score = _evaluator(train_pred[_train_idx], _labels[_train_idx])

    if args.sample_type == "khop_sample":

        if args.use_labels:
            global_train_idx = np.random.permutation(_train_idx.cpu().clone())
            global_labels_idx = global_train_idx[:int(len(global_train_idx)*args.mask_rate)]
            global_pred_idx = global_train_idx[int(len(global_train_idx)*args.mask_rate):]

        for nodes, root_nodes, subgraph in dataloader:
            subgraph = subgraph.to(device)
            new_train_idx = root_nodes
            
            if args.use_labels:
                train_labels_idx = new_train_idx[np.isin(nodes[new_train_idx], global_labels_idx)]
                train_pred_idx = new_train_idx[np.isin(nodes[new_train_idx], global_pred_idx)]

                add_labels(subgraph, train_labels_idx, n_classes, device)
            else:
                train_pred_idx = new_train_idx
            
            pred = model(subgraph)
            loss = criterion(pred[train_pred_idx], subgraph.ndata["labels"][train_pred_idx].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = len(train_pred_idx)
            loss_sum += loss.item() * count
            total += count

    if args.sample_type == "m_cluster":
        for subgraph in dataloader:
            subgraph = subgraph.to(device)
            pred = model(subgraph)
            mask = subgraph.ndata["train_mask"]
            label = subgraph.ndata["labels"].float()
            # print(pred.size())
            # print(torch.count_nonzero(mask).item())
            # print(pred[mask].size())
            loss = criterion(pred[mask], label[mask].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = torch.count_nonzero(subgraph.ndata["train_mask"]).item()
            loss_sum += loss.item() * count
            total += count

    return loss_sum / total, train_score


@torch.no_grad()
def evaluate(args, graph, model, dataloader, labels, train_idx, val_idx, test_idx, criterion, evaluator):
    torch.cuda.empty_cache()
    model.eval()

    preds = torch.zeros(labels.shape).to(device)

    # Due to the memory capacity constraints, we use sampling for inference and calculate the average of the predictions 'eval_times' times.

    for _ in range(args.eval_times):

        if args.sample_type == "neighbor_sample":
            for input_nodes, output_nodes, subgraphs in dataloader:
                subgraphs = [b.to(device) for b in subgraphs]
                new_train_idx = list(range(len(input_nodes)))

                if args.use_labels:
                    add_labels(subgraphs[0], new_train_idx, n_classes, device)

                pred = model(subgraphs)
                preds[output_nodes] += pred

        if args.sample_type in ["random_cluster", "khop_sample"]:
            for batch_nodes, subgraph in random_partition_v2(args.eval_partition_num, graph, shuffle=False):
                subgraph = subgraph.to(device)
                new_train_idx = torch.arange(len(batch_nodes))
                # label_idx = new_train_idx[np.isin(batch_nodes, train_idx.cpu())]

                if args.use_labels:
                    add_labels(subgraph, new_train_idx, n_classes, device)

                pred = model(subgraph)
                preds[batch_nodes] += pred

        if args.sample_type == "m_cluster":
            for subgraph in dataloader:
                subgraph = subgraph.to(device)

                if args.use_labels:
                    add_labels(subgraph, subgraph.ndata["train_mask"], n_classes, device)

                pred = model(subgraph)
                preds[subgraph.ndata["NID"]] += pred

    preds /= args.eval_times

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
        [preds[train_idx], preds[val_idx], preds[test_idx]],
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running, log_f):
    evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred, "y_true": labels})["rocauc"]

    train_dataloader, eval_dataloader = None, None

    if args.sample_type == "neighbor_sample":
        real_layer = args.n_layers if args.model != "agdn" else args.n_layers * args.K
        train_batch_size = (len(train_idx) + args.train_partition_num - 1) // args.train_partition_num
        train_sampler = MultiLayerNeighborSampler([32 for _ in range(real_layer)])
        train_dataloader = dgl.dataloading.DataLoader(graph.cpu(), train_idx.cpu(),
            train_sampler, batch_size=train_batch_size, shuffle=True, num_workers=8)

        eval_batch_size = (len(labels) + args.eval_partition_num - 1) // args.eval_partition_num
        eval_sampler = MultiLayerNeighborSampler([100 for _ in range(real_layer)])
        eval_dataloader = dgl.dataloading.DataLoader(
                                            graph.cpu(),torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]),
                                            eval_sampler, batch_size=eval_batch_size, shuffle=True, num_workers=8)

    if args.sample_type == "khop_sample":
        train_batch_size = (len(train_idx) + args.train_partition_num - 1) // args.train_partition_num
        train_dataloader = ShaDowKHopSampler(graph.cpu(), 
                                             args.sampler_K, 
                                             args.sampler_budget, 
                                             train_idx.cpu(), 
                                             replace=False, 
                                             num_workers=10,
                                             batch_size=train_batch_size, 
                                             shuffle=True)
        eval_batch_size = (len(labels) + args.eval_partition_num - 1) // args.eval_partition_num
        eval_dataloader = ShaDowKHopSampler(graph.cpu(), 
                                            args.sampler_K, 
                                            args.sampler_budget, 
                                            torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]), 
                                            replace=False, 
                                            num_workers=10,
                                            batch_size=eval_batch_size, 
                                            shuffle=False)

    if args.sample_type == "m_cluster":
        t_path = args.root + "ogbn_proteins/cluster/cluster_%d_%d.pkl" %(args.train_partition_num, n_running)
        e_path = args.root + "ogbn_proteins/processed/cluster_%d_%d.pkl" % (args.eval_partition_num, n_running) #int(time.time())
        train_sampler = dgl.dataloading.ClusterGCNSampler(graph, args.train_partition_num, cache_path=t_path)
        eval_sampler =  dgl.dataloading.ClusterGCNSampler(graph, args.eval_partition_num, cache_path=e_path)
        train_dataloader = dgl.dataloading.DataLoader(graph.cpu(), torch.arange(args.train_partition_num), train_sampler,
                                                      batch_size=args.sampler_K, shuffle=True, drop_last=False, num_workers=8)
        eval_dataloader = dgl.dataloading.DataLoader(graph.cpu(), torch.arange(args.eval_partition_num), eval_sampler,
                                                      batch_size=1, shuffle=False, drop_last=False, num_workers=8)

    criterion = nn.BCEWithLogitsLoss()

    model = gen_model(args, n_node_feats, n_edge_feats, n_classes).to(device)

    lr_scheduler = None
    if args.advanced_optimizer:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=50, verbose=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    

    total_time = 0
    val_score, best_val_score, final_test_score, best_step = 0, 0, 0, 0

    train_scores, val_scores, test_scores = [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []
    final_pred = None


    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        loss, t_score = train(args, graph, model, train_dataloader, labels, train_idx, val_idx, test_idx, criterion,
                         optimizer, evaluator_wrapper)

        toc = time.time()
        total_time += toc - tic
        train_msg = f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, " \
                    f"this epoch time: {toc - tic:.2f}s Train loss/score: {loss:.4f}/{t_score:.4f}\n"
        print_msg_and_write(train_msg, log_f)

        if epoch < 500:
            eval_interval, log_interval = args.eval_every * 20, args.log_every * 20
        elif epoch < 1000:
            eval_interval, log_interval = args.eval_every * 10, args.log_every * 10
        elif epoch > 1100:
            eval_interval, log_interval =  args.eval_every * 5, args.log_every * 5
        else:
            eval_interval, log_interval = args.eval_every * 2, args.log_every * 2

        if epoch == args.n_epochs or epoch % eval_interval== 0 or epoch % log_interval == 0:
            train_score, val_score, test_score, train_loss, val_loss, test_loss, pred = evaluate(
                args, graph, model, eval_dataloader, labels, train_idx, val_idx, test_idx, criterion, evaluator_wrapper)

            if val_score > best_val_score:
                best_val_score = val_score
                final_test_score = test_score
                final_pred = pred
                best_step = epoch

            if epoch % log_interval == 0:
                out_msg = f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}s\n" \
                        f"Loss: {loss:.4f} Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n" \
                        f"Train/Val/Test: {train_score:.4f}/{val_score:.4f}/{test_score:.4f}\n"\
                        f"Best val/Final test score/Best Step: {best_val_score:.4f}/{final_test_score:.4f}/{best_step}\n"
                print_msg_and_write(out_msg, log_f)
            for l, e in zip(
                [train_scores, val_scores, test_scores, losses, train_losses, val_losses, test_losses],
                [train_score, val_score, test_score, loss, train_loss, val_loss, test_loss],
            ):
                l.append(e)

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

    argparser = argparse.ArgumentParser(
        "GAT implementation on ogbn-proteins", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
    argparser.add_argument("--edge-emb-size", type=int, default=16)
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

    # if args.use_labels:
    #     n_node_feats += 2 * n_classes

    labels, train_idx, val_idx, test_idx = map(lambda x: x.to(device), (labels, train_idx, val_idx, test_idx))

    # run
    val_scores, test_scores = [], []
    title_msg = f"Number of node feature: {n_node_feats}\n" + f"Number of edge feature: {n_edge_feats}\n" + \
                f"Number of params: {count_parameters(args, n_node_feats, n_edge_feats, n_classes)}\n"
    print(title_msg)
    version = str(int(time.time())) if args.log_file_name=="" else "%s_%d" %(args.log_file_name, int(time.time()))
    os.makedirs("%s/log" % (args.root), exist_ok=True)
    for i in range(args.n_runs):
        log_f = open("%s/log/%s_part%d.log" % (args.root, version, i) , mode='a')
        log_f.write(args.__str__() + "\n" + title_msg)
        log_f.flush()
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


