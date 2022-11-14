from data import load_data, preprocess
root_path = "/Users/lihouyi/Documents/opensource/data_source/proteins"
dataset = "ogbn-proteins"
n_node_feats, n_edge_feats, n_classes = 0, 8, 112
graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, root_path)
graph, labels = preprocess(graph, labels, train_idx, n_classes, val_idx=val_idx, test_idx=test_idx)

print("Preprocessing")