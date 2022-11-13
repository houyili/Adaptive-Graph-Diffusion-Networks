from models import AGDN

class GIPA(AGDN):
    def __init__(
            self,
            node_feats,
            edge_feats,
            n_classes,
            n_layers,
            n_heads,
            n_hidden,
            edge_emb,
            activation,
            dropout,
            input_drop,
            attn_drop,
            hop_attn_drop,
            edge_drop,
            K=3,
            use_attn_dst=True,
            allow_zero_in_degree=False,
            norm="none",
            use_one_hot=False,
            use_labels=False,
            edge_attention=False,
            weight_style="HA", batch_norm=True, edge_att_act="leaky_relu", edge_agg_mode="both_softmax"
    ):
        super(GIPA, self).__init__(node_feats, edge_feats,
            n_classes, n_layers, n_heads, n_hidden, edge_emb, activation, dropout, input_drop,
            attn_drop, hop_attn_drop, edge_drop, K, use_attn_dst, allow_zero_in_degree,
            norm, use_one_hot, use_labels, edge_attention,
            weight_style, batch_norm, edge_att_act, edge_agg_mode)
