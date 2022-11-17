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
graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, root_path) # 132534node 79122504edge
# graph, labels = preprocess(graph, labels, train_idx)

seed_nodes = torch.arange(10, 20)
frontier = graph.sample_neighbors(seed_nodes, -1) # 132534node 79122504edge

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

import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
# root_path = "/data/ogb/datasets/"
root_path = "/Users/lihouyi/Documents/opensource/data_source/proteins"
dataset = "ogbn-proteins"
data = DglNodePropPredDataset(name=dataset, root=root_path)
graph, labels = data[0]
seed_nodes = torch.arange(10, 20)
frontier = graph.sample_neighbors(seed_nodes, -1) # 132534node 79122504edgeb

frontier.all_edges()[0].size() #是dstnode id，
frontier.all_edges()[1].max() # 是 srcnode id 原有的
frontier.dstnodes() #frontier.srcnodes() 里面有全部的点
frontier.dstdata['species'].size() #torch.Size([132534, 1])
# frontier.dstdata['labels'].size() #torch.Size([132534, 112])
frontier.edata['feat'].size() #torch.Size([3396, 8])
frontier.edata['_ID'].size() #torch.Size([3396])
frontier.edata['_ID'].max() #tensor(26950967)

frontier_src_node = frontier.all_edges()[1]
frontier_dst_node = frontier.all_edges()[0]
frontier.in_degree(seed_nodes) #  tensor([  80,  363,   96,  219,   59, 1363,  250,   67,  353,  546])
frontier.out_degree(seed_nodes) # tensor([0, 0, 1, 0, 0, 0, 0, 0, 1, 0])
in_loop_index = torch.nonzero(torch.isin(frontier_dst_node, seed_nodes)) # [ 538], [2614]
frontier_dst_node[in_loop_index].int() #  [[18], [12]]
in_loop_edge_id = frontier.edata['_ID'][in_loop_index] # [[21649991], [21649990]]


block = dgl.to_block(frontier, seed_nodes)
block.srcdata['_ID'].size() # 原有的 #tensor([  10,   11,   12,  ..., 6527, 6528, 6529])
block.srcdata['_ID'].max() # tensor(125423)
block.dstdata['_ID'].size()
block.dstdata['_ID'].max()
block.edata['_ID'].max() # tensor(3395)
block.srcnodes().size() # replace
block.srcnodes().max()
block.dstnodes().size()
block.dstnodes().max()
block.all_edges()[0].max()
block.all_edges()[1].max()
block.all_edges()[0].size() # replace
block.all_edges()[1].size()

a = frontier.subgraph(torch.cat([seed_nodes, frontier_src_node[frontier_src_node % 3 == 1]]))
a.srcdata['_ID'].max() # 里面是原有的ID，
a.dstdata['_ID'].size() # torch.Size([1129])
a.dstdata['_ID'].max()
a.srcnodes()
a.dstnodes()  # tensor([   0,    1,    2,  ..., 1126, 1127, 1128])
a.all_edges() # 里面的id都换掉了，从0开始
a.edata['_ID'].max() # 里面的id 也换掉了， 从9开始

# bl_a = dgl.to_block(a, seed_nodes) # 不能用

random_num = torch.rand(frontier.edata['_ID'].size())
edge_id = torch.arange(frontier.num_edges())
random_edges = edge_id[random_num < 0.2]
b = dgl.edge_subgraph(frontier, random_edges, relabel_nodes=False, store_ids=False)
b.all_edges()[0].max() # tensor(123929)
b.all_edges()[0].size() # torch.Size([1000])
b.all_edges()[1].max() # 是 srcnode id 原有的
b.edata['feat'].size() #torch.Size([3396, 8])
b.edata['_ID'].size()  #torch.Size([681])
b.edata['_ID'].max()   #tensor(3393)
b.dstnodes() #frontier.srcnodes() 里面有全部的点
b.dstdata['species'].size() #torch.Size([132534, 1])
b.srcdata['species'].size() #torch.Size([132534, 1])

b_src_node = b.all_edges()[1]
b_dst_node = b.all_edges()[0]
b_indegree = b.in_degrees(seed_nodes) # tensor([ 15,  67,  24,  40,  14, 277,  55,  17,  66, 106])
b_outdegree = b.out_degrees(seed_nodes) # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# 取出seed node 在随机图中degree小于15的点  seed_nodes[b_indegree < 15]
# 取出代表 原子图中的index torch.nonzero(frontier.all_edges()[1] == seed_nodes[b_indegree < 15])
index_in_frontier = torch.nonzero(torch.isin(frontier.all_edges()[1], seed_nodes[b_indegree < 15]))
random_num[index_in_frontier] = 0
b2 = dgl.edge_subgraph(frontier, edge_id[random_num < 0.2], relabel_nodes=False, store_ids=False)
b2.in_degrees(seed_nodes) # tensor([ 15,  67,  24,  40,  59, 277,  55,  17,  66, 106])


bl_b = dgl.to_block(b, seed_nodes)
bl_b.srcdata['_ID'].size() # torch.Size([987])
bl_b.srcdata['_ID'].max() # tensor(123929)
bl_b.dstdata['_ID'].size() # torch.Size([10])
bl_b.dstdata['_ID'].max() # tensor(19)
bl_b.srcnodes().size() # replace
bl_b.srcnodes().max() # tensor(986)
bl_b.dstnodes().size() # torch.Size([10])
bl_b.dstnodes().max()
bl_b.edata['_ID'].size() # torch.Size([1000])
bl_b.edata['_ID'].max() # tensor(3395)
bl_b.all_edges()[0].max() # tensor(986)
bl_b.all_edges()[1].max() # tensor(9)
bl_b.all_edges()[0].size() # torch.Size([1000])
bl_b.all_edges()[1].size()



from new_sample import EdgeSampleNeighborSampler, InSeedNodeNeighborSampler
import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
root_path = "/data/ogb/datasets/"
# root_path = "/Users/lihouyi/Documents/opensource/data_source/proteins"
dataset = "ogbn-proteins"
data = DglNodePropPredDataset(name=dataset, root=root_path)
graph, labels = data[0]
seed_nodes = torch.arange(10, 13000)
train_sampler = EdgeSampleNeighborSampler([-1, -1, -1, -1, -1, -1], 0.2)
frontier = train_sampler.sample_blocks(graph, seed_nodes)

train_sampler_2 = dgl.dataloading.MultiLayerNeighborSampler([-1 for _ in range(6)])
frontier_2 = train_sampler_2.sample_blocks(graph, seed_nodes)

train_sampler_3 = dgl.dataloading.MultiLayerNeighborSampler([32, 32, 32, 100 ,100 ,100])
frontier_3 = train_sampler_3.sample_blocks(graph, seed_nodes)


splitted_idx = data.get_idx_split()
train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
train_sampler = InSeedNodeNeighborSampler([-1, -1, -1, -1, -1, -1])
seed_nodes_set = torch.randperm(len(train_idx))
seed = seed_nodes_set[:len(train_idx)/6]
src_nodes, dst_nodes, blocks = train_sampler.sample_blocks(graph, seed)

