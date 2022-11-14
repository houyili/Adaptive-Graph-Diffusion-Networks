import dgl.function as fn
import torch
import numpy as np
from sklearn import preprocessing
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from utils import compute_norm

def load_data(dataset, root_path):
    data = DglNodePropPredDataset(name=dataset, root=root_path)
    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    graph.ndata["labels"] = labels
    print(f"Nodes : {graph.number_of_nodes()}\n"
          f"Edges: {graph.number_of_edges()}\n"
          f"Train nodes: {len(train_idx)}\n"
          f"Val nodes: {len(val_idx)}\n"
          f"Test nodes: {len(test_idx)}")

    return graph, labels, train_idx, val_idx, test_idx, evaluator

def preprocess(graph, labels, train_idx, n_classes, edge_agg_as_feat=True, one_hot_feat=True, user_label=True,
               user_adj=True, user_avg=True, val_idx=None, test_idx=None):

    graph.ndata["NID"] = torch.arange(graph.num_nodes())
    train_mask = torch.zeros(size=(graph.num_nodes(),1), dtype=torch.bool)
    train_mask[train_idx] = True
    graph.ndata["train_mask"] = train_mask
    if val_idx != None:
        val_mask = torch.zeros(size=(graph.num_nodes(),1), dtype=torch.bool)
        val_mask[val_idx] = True
        graph.ndata["val_mask"] = val_mask

    if test_idx != None:
        test_mask = torch.zeros(size=(graph.num_nodes(),1), dtype=torch.bool)
        test_mask[test_idx] = True
        graph.ndata["test_mask"] = test_mask


    # The sum of the weights of adjacent edges is used as node features.
    if edge_agg_as_feat:
        graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))

    if one_hot_feat:
        le = preprocessing.LabelEncoder()
        species_unique = torch.unique(graph.ndata["species"])
        max_no = species_unique.max()
        le.fit(species_unique % max_no)
        species = le.transform(graph.ndata["species"].squeeze() % max_no)
        species = np.expand_dims(species, axis=1)

        enc = preprocessing.OneHotEncoder()
        enc.fit(species)
        one_hot_encoding = enc.transform(species).toarray()

        graph.ndata["x"] = torch.FloatTensor(one_hot_encoding)

    if user_label:
        # Only the labels in the training set are used as features, while others are filled with zeros.
        graph.ndata["train_labels_onehot"] = torch.zeros(graph.number_of_nodes(), n_classes)
        graph.ndata["train_labels_onehot"][train_idx] = labels[train_idx].float()
        graph.ndata["deg"] = graph.out_degrees().float().clamp(min=1)

    if user_adj or user_avg:
        deg_sqrt, deg_isqrt = compute_norm(graph)
        if user_adj:
            graph.srcdata.update({"src_norm": deg_isqrt})
            graph.dstdata.update({"dst_norm": deg_sqrt})
            graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm_adjust"))

        if user_avg:
            graph.srcdata.update({"src_norm": deg_isqrt})
            graph.dstdata.update({"dst_norm": deg_isqrt})
            graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm"))

    graph.create_formats_()
    print(graph.ndata.keys())
    print(graph.edata.keys())

    return graph, labels
