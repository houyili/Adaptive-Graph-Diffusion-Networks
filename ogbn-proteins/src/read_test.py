import torch
import dgl
from dgl.dataloading import MultiLayerNeighborSampler, DataLoader

root_path = "/Users/lihouyi/Documents/opensource/data_source/proteins"
dataset = "ogbn-proteins"
n_node_feats, n_edge_feats, n_classes = 0, 8, 112

# from data import load_data, preprocess
# graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, root_path)
# graph, labels = preprocess(graph, labels, train_idx, n_classes, val_idx=val_idx, test_idx=test_idx)

from main_gat_2 import load_data, preprocess, add_labels
graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, root_path)
graph, labels = preprocess(graph, labels, train_idx)

train_sampler = MultiLayerNeighborSampler([32 for _ in range(6)])
train_dataloader = DataLoader(graph.cpu(), train_idx.cpu(), train_sampler, batch_size=1000)
for input_nodes, output_nodes, subgraphs in train_dataloader:
    new_train_idx = torch.arange(len(output_nodes))
    train_labels_idx = torch.arange(len(output_nodes), len(input_nodes))
    train_pred_idx = new_train_idx
    add_labels(subgraphs[0], train_labels_idx)
    len(output_nodes)

print("Preprocessing")
cluster_sampler = dgl.dataloading.ClusterGCNSampler(graph, 50, cache_path=root_path+"/processed/cluster_gcn.pkl")
node0 = torch.LongTensor([0,3])
sg = cluster_sampler.sample(graph, node0)

print("Preprocessing")
