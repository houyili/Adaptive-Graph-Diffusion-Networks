import numpy as np
import torch.nn.functional as F

from final_model import GIPADeep, GIPAPara
from models import GAT, AGDN
from new_models import AGDN_MA, AGDN_SM, GIPASMConv, AGDNSMConv

MODEL_LIST = ["gat", "agdn", "agdn_ma", "agdn_sm", "gipa_sm", "gipa_deep", "gipa_para"]

def gen_model(args, n_node_feats, n_edge_feats, n_classes):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats

    model = None
    if args.model == "gat":
        model = GAT(
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
            edge_drop=args.edge_drop,
            use_attn_dst=not args.no_attn_dst,
            norm=args.norm,
            use_one_hot_feature=args.use_one_hot_feature,
            use_labels=args.use_labels,
        )

    if args.model == "agdn":
        model = AGDN(
            n_node_feats_,
            n_edge_feats,
            n_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_hidden=args.n_hidden,
            edge_emb=args.edge_emb_size,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            hop_attn_drop=args.hop_attn_drop,
            edge_drop=args.edge_drop,
            K=args.K,
            use_attn_dst=not args.no_attn_dst,
            norm=args.norm,
            use_one_hot=args.use_one_hot_feature,
            use_labels=args.use_labels,
            weight_style=args.weight_style,
            batch_norm=not args.disable_fea_trans_norm,
            edge_att_act=args.edge_att_act, edge_agg_mode=args.edge_agg_mode
        )

    if args.model == "agdn_ma":
        model = AGDN_MA(
            n_node_feats_,
            n_edge_feats,
            n_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_hidden=args.n_hidden,
            edge_emb=args.edge_emb_size,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            hop_attn_drop=args.hop_attn_drop,
            edge_drop=args.edge_drop,
            K=args.K,
            use_attn_dst=not args.no_attn_dst,
            norm=args.norm,
            use_one_hot=args.use_one_hot_feature,
            use_labels=args.use_labels,
            weight_style=args.weight_style,
            batch_norm=not args.disable_fea_trans_norm,
            edge_att_act=args.edge_att_act, edge_agg_mode=args.edge_agg_mode
        )

    if args.model == "agdn_sm" or args.model == "gipa_sm":
        kernel = AGDNSMConv if args.model == "agdn_sm" else GIPASMConv
        model = AGDN_SM(
            n_node_feats_,
            n_edge_feats,
            n_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_hidden=args.n_hidden,
            edge_emb=args.edge_emb_size,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            hop_attn_drop=args.hop_attn_drop,
            edge_drop=args.edge_drop,
            K=args.K,
            use_attn_dst=not args.no_attn_dst,
            norm=args.norm,
            use_one_hot=args.use_one_hot_feature,
            use_labels=args.use_labels,
            weight_style=args.weight_style,
            batch_norm=not args.disable_fea_trans_norm,
            edge_att_act=args.edge_att_act, edge_agg_mode=args.edge_agg_mode,
            conv_kernel=kernel
        )

    if args.model == "gipa_deep":
        model = GIPADeep(
            n_node_feats_,
            n_edge_feats,
            n_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_hidden=args.n_hidden,
            edge_emb=args.edge_emb_size,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            edge_drop=args.edge_drop,
            use_attn_dst=not args.no_attn_dst,
            norm=args.norm,
            use_one_hot=args.use_one_hot_feature,
            batch_norm=not args.disable_fea_trans_norm,
            edge_att_act=args.edge_att_act, edge_agg_mode=args.edge_agg_mode,
            first_hidden=args.first_hidden,
            use_att_edge= not args.disable_att_edge,
            use_prop_edge=args.use_prop_edge
        )

    if args.model == "gipa_para":
        model = GIPAPara(
            n_node_feats_,
            n_edge_feats,
            n_classes,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_hidden=args.n_hidden,
            edge_emb=args.edge_emb_size,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            edge_drop=args.edge_drop,
            use_attn_dst=not args.no_attn_dst,
            norm=args.norm,
            use_one_hot=args.use_one_hot_feature,
            batch_norm=not args.disable_fea_trans_norm,
            edge_att_act=args.edge_att_act, edge_agg_mode=args.edge_agg_mode,
            first_hidden=args.first_hidden,
            use_att_edge= not args.disable_att_edge,
            use_prop_edge=args.use_prop_edge,
            n_hidden_per_head=args.n_hidden_per_head
        )
    return model


def count_parameters(args, n_node_feats, n_edge_feats, n_classes):
    model = gen_model(args, n_node_feats, n_edge_feats, n_classes)
    for p in model.parameters():
        if p.requires_grad:
            print(p.__)
            print(p.size())
    n_parameters = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    del model
    return n_parameters
