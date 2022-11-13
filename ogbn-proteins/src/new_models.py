from models import AGDN, AGDNConv, EdgeAttentionLayer
import torch.nn as nn

class AGDN_MA(AGDN):
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
        # super(GIPA, self).__init__(node_feats, edge_feats,
        #     n_classes, n_layers, n_heads, n_hidden, edge_emb, activation, dropout, input_drop,
        #     attn_drop, hop_attn_drop, edge_drop, K, use_attn_dst, allow_zero_in_degree,
        #     norm, use_one_hot, use_labels, edge_attention,
        #     weight_style, batch_norm, edge_att_act, edge_agg_mode)
        super(AGDN, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_encoder = nn.Linear(node_feats, n_heads)
        if edge_attention:
            self.pre_aggregator = EdgeAttentionLayer(edge_feats, n_heads)
        else:
            self.pre_aggregator = None
        if use_one_hot:
            self.one_hot_encoder = nn.Linear(8, 8)
        else:
            self.one_hot_encoder = None

        if edge_emb > 0:
            self.edge_encoder = nn.ModuleList()
            self.edge_norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else n_heads
            out_hidden = n_hidden
            # bias = i == n_layers - 1

            if edge_emb > 0:
                self.edge_encoder.append(nn.Linear(edge_feats, edge_emb))
                self.edge_norms.append(nn.BatchNorm1d(edge_emb))
            self.convs.append(
                AGDNConv(
                    in_hidden,
                    edge_emb,
                    out_hidden,
                    n_heads=n_heads,
                    K=K,
                    attn_drop=attn_drop,
                    hop_attn_drop=hop_attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    residual=True,
                    allow_zero_in_degree=allow_zero_in_degree,
                    norm=norm,
                    weight_style=weight_style, batch_norm=batch_norm, edge_att_act=edge_att_act,
                    edge_agg_mode=edge_agg_mode
                )
            )
            self.norms.append(nn.BatchNorm1d(n_heads * out_hidden))

        self.pred_linear = nn.Linear(n_heads * n_hidden, n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation