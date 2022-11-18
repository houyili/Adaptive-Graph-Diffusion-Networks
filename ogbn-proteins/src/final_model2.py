import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair

from new_models import GIPASMConv
from utils import get_act_by_str

class GIPA2Conv(nn.Module):
    def __init__(
            self,
            node_feats,
            edge_feats,
            out_feats,
            n_heads=1,
            edge_drop=0.0,
            negative_slope=0.2,
            activation=None,
            use_attn_dst=True,
            allow_zero_in_degree=True,
            norm="none",
            batch_norm=True,
            weight_style="HA",
            edge_att_act="leaky_relu",
            edge_agg_mode="both_softmax",
            use_att_edge=True,
            use_prop_edge=False
    ):
        super(GIPA2Conv, self).__init__()
        self._n_heads = n_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._norm = norm
        self._batch_norm = batch_norm
        self._weight_style = weight_style
        self._edge_agg_mode = edge_agg_mode
        self._edge_att_act = edge_att_act

        # optional fc
        self.prop_edge_fc = None
        self.attn_dst_fc = None
        self.attn_edge_fc = None

        # propagation src feature
        self.src_fc = nn.Linear(self._in_src_feats, out_feats, bias=False)
        if use_prop_edge:
            self.prop_edge_fc = nn.Linear(edge_feats, out_feats, bias=False)

        # apply function
        self.dst_fc = nn.Linear(self._in_src_feats, out_feats)

        # attn fc
        self.attn_src_fc = nn.Linear(self._in_src_feats, out_feats, bias=False)
        if use_attn_dst:
            self.attn_dst_fc = nn.Linear(self._in_src_feats, out_feats, bias=False)
        if edge_feats > 0 and use_att_edge:
            self.attn_edge_fc = nn.Linear(edge_feats, out_feats, bias=False)
            self.edge_norm = nn.BatchNorm1d(edge_feats)

        if batch_norm:
            self.offset, self.scale = nn.ParameterList(), nn.ParameterList()
            self.offset.append(nn.Parameter(torch.zeros(size=(1, n_heads, out_feats//n_heads))))
            self.scale.append(nn.Parameter(torch.ones(size=(1, n_heads, out_feats//n_heads))))

        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
        self.edge_att_actv = get_act_by_str(edge_att_act, negative_slope)
        self.activation = activation
        self.agg_fc = nn.Linear(out_feats, out_feats)

        print("Init %s" % str(self.__class__))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.src_fc.weight, gain=gain)
        if self.dst_fc is not None:
            nn.init.xavier_normal_(self.dst_fc.weight, gain=gain)
        if self.prop_edge_fc is not None:
            nn.init.xavier_normal_(self.prop_edge_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
        # nn.init.zeros_(self.attn_src_fc.bias)
        if self.attn_dst_fc is not None:
            nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)
            # nn.init.zeros_(self.attn_dst_fc.bias)
        if self.attn_edge_fc is not None:
            nn.init.xavier_normal_(self.attn_edge_fc.weight, gain=gain)
            # nn.init.zeros_(self.attn_edge_fc.bias)
        nn.init.xavier_normal_(self.agg_fc.weight, gain=gain)

    def agg_function(self, h, idx):
        if self._batch_norm:
            h = h.view(-1, self._n_heads, self._out_feats//self._n_heads)
            mean = h.mean(dim=-1).view(h.shape[0], self._n_heads, 1)
            var = h.var(dim=-1, unbiased=False).view(h.shape[0], self._n_heads, 1) + 1e-9
            h = (h - mean) * self.scale[idx] * torch.rsqrt(var) + self.offset[idx]
        return self.agg_fc(h.view(-1, self._out_feats))

    def forward(self, graph, feat_src, feat_edge=None):
        with graph.local_scope():
            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:
                feat_dst = feat_src

            # propagation value prepare
            feat_src_fc = self.src_fc(feat_src)
            graph.srcdata.update({"_feat_src_fc": feat_src_fc})
            if self.prop_edge_fc is not None and feat_edge is not None:
                graph.edata["_v"]  = self.prop_edge_fc(feat_edge)
                graph.apply_edges(fn.u_add_e("_feat_src_fc", "_v", "_prop_edge"))

            # src node attention
            attn_src = self.attn_src_fc(feat_src)
            graph.srcdata.update({"_attn_src": attn_src})

            # dst node attention
            if self.attn_dst_fc is not None:
                attn_dst = self.attn_dst_fc(feat_dst)
                graph.dstdata.update({"_attn_dst": attn_dst})
                graph.apply_edges(fn.u_add_v("_attn_src", "_attn_dst", "_attn_node"))
            else:
                graph.apply_edges(fn.copy_u("_attn_src", "_attn_node"))

            e = graph.edata["_attn_node"]
            if self.attn_edge_fc is not None:
                attn_edge = self.attn_edge_fc(feat_edge)
                graph.edata.update({"_attn_edge": attn_edge})
                e += graph.edata["_attn_edge"]

            e = self.edge_att_actv(e)

            if self._edge_agg_mode == "both_softmax":
                graph.edata["_a"] = torch.sqrt(edge_softmax(graph, e,  norm_by='dst').clamp(min=1e-9)
                    * edge_softmax(graph, e,  norm_by='src').clamp(min=1e-9))
            elif self._edge_agg_mode == "single_softmax":
                graph.edata["_a"] = edge_softmax(graph, e, norm_by='dst')
            else:
                graph.edata["_a"] = e

            if self._norm == "adj":
                graph.edata["_a"] = graph.edata["_a"] * graph.edata["gcn_norm_adjust"].view(-1, 1)
            if self._norm == "avg":
                graph.edata["_a"] = (graph.edata["_a"] * graph.edata["gcn_norm"].view(-1, 1)) / 2

            if self.prop_edge_fc is not None and feat_edge is not None:
                graph.edata["_m"] = graph.edata["_a"] * graph.edata["_prop_edge"]
                graph.update_all(fn.copy_e("_m", "_m_copy"), fn.sum("_m_copy", "_feat_src_fc"))
            else:
                graph.update_all(fn.u_mul_e("_feat_src_fc", "_a", "_m"), fn.sum("_m", "_feat_src_fc"))
            msg_sum = graph.dstdata["_feat_src_fc"]
            # print(msg_sum.size())
            # aggregation function
            rst = self.agg_function(msg_sum, 0)

            # apply function
            if self.dst_fc is not None:
                rst += self.dst_fc(feat_dst)
            if self.activation is not None:
                rst = self.activation(rst, inplace=True)
            return rst

class GIPA2Deep(nn.Module):
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
            edge_drop,
            use_attn_dst=True,
            allow_zero_in_degree=False,
            norm="none",
            use_one_hot=False,
            batch_norm=True, edge_att_act="leaky_relu", edge_agg_mode="both_softmax",
            first_hidden = 150,
            use_att_edge=True,
            use_prop_edge=False
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_encoder = nn.Linear(node_feats, first_hidden)
        if use_one_hot:
            self.one_hot_encoder = nn.Linear(8, 8)
        else:
            self.one_hot_encoder = None

        if edge_emb > 0:
            self.edge_encoder = nn.ModuleList()
            self.edge_norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden =  n_hidden if i > 0 else first_hidden
            out_hidden = n_hidden

            if edge_emb > 0:
                self.edge_encoder.append(nn.Linear(edge_feats, edge_emb))
                self.edge_norms.append(nn.BatchNorm1d(edge_emb))
            self.convs.append(
                GIPA2Conv(
                    in_hidden,
                    edge_emb,
                    out_hidden,
                    n_heads=n_heads,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    allow_zero_in_degree=allow_zero_in_degree,
                    norm=norm,
                    batch_norm=batch_norm, edge_att_act=edge_att_act,
                    edge_agg_mode=edge_agg_mode,
                    use_att_edge=use_att_edge,
                    use_prop_edge=use_prop_edge
                )
            )
            self.norms.append(nn.BatchNorm1d(out_hidden))

        self.pred_linear = nn.Linear(n_hidden, n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        print("The new parameter are %s,%s,%s" % (batch_norm, edge_att_act, edge_agg_mode))
        print("Init %s" % str(self.__class__))

    def forward(self, g):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g

        h = subgraphs[0].srcdata["feat"]

        if self.one_hot_encoder is not None:
            x = subgraphs[0].srcdata["x"]
            h = torch.cat([x, h], dim=1)

        h = self.node_encoder(h)
        h = F.relu(h, inplace=True)
        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):

            if self.edge_encoder is not None:
                efeat = subgraphs[i].edata["feat"]
                efeat_emb = self.edge_encoder[i](efeat)
                efeat_emb = F.relu(efeat_emb, inplace=True)
            else:
                efeat_emb = None

            h = self.convs[i](subgraphs[i], h, efeat_emb).flatten(1, -1)

            if h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h
            h = self.norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout(h)

        h = self.pred_linear(h)
        return h

class GIPA2Para(nn.Module):
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
            edge_drop,
            use_attn_dst=True,
            allow_zero_in_degree=False,
            norm="none",
            use_one_hot=False,
            batch_norm=True, edge_att_act="leaky_relu", edge_agg_mode="both_softmax",
            first_hidden = 150,
            use_att_edge=True,
            use_prop_edge=False,
            n_hidden_per_head=30
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.norms2 = nn.ModuleList()

        self.node_encoder = nn.Linear(node_feats, first_hidden)
        self.node_encoder2 = nn.Linear(node_feats, first_hidden)
        if use_one_hot:
            self.one_hot_encoder = nn.Linear(8, 8)
        else:
            self.one_hot_encoder = None

        if edge_emb > 0:
            self.edge_encoder = nn.ModuleList()
            self.edge_encoder2 = nn.ModuleList()

        for i in range(n_layers):
            in_hidden =  n_hidden if i > 0 else first_hidden
            in_hidden2 = n_heads * n_hidden_per_head if i > 0 else first_hidden
            out_hidden = n_hidden
            out_hidden2 =  n_heads * n_hidden_per_head

            if edge_emb > 0:
                self.edge_encoder.append(nn.Linear(edge_feats, edge_emb))
                self.edge_encoder2.append(nn.Linear(edge_feats, edge_emb))

            self.convs.append(
                GIPA2Conv(
                    in_hidden,
                    edge_emb,
                    out_hidden,
                    n_heads=n_heads,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    allow_zero_in_degree=allow_zero_in_degree,
                    norm=norm,
                    batch_norm=batch_norm, edge_att_act=edge_att_act,
                    edge_agg_mode=edge_agg_mode,
                    use_att_edge=use_att_edge,
                    use_prop_edge=use_prop_edge
                )
            )
            self.convs2.append(
                GIPASMConv(
                    in_hidden2,
                    edge_emb,
                    n_hidden_per_head,
                    n_heads=n_heads,
                    K=1,
                    attn_drop=0.0,
                    hop_attn_drop=0.0,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    residual=True,
                    allow_zero_in_degree=allow_zero_in_degree,
                    norm=norm,
                    weight_style="sum", batch_norm=batch_norm, edge_att_act=edge_att_act,
                    edge_agg_mode=edge_agg_mode
                )
            )
            self.norms.append(nn.BatchNorm1d(out_hidden))
            self.norms2.append(nn.BatchNorm1d(out_hidden2))

        self.pred_linear = nn.Linear(n_hidden + n_heads * n_hidden_per_head, n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        print("Init %s" % str(self.__class__))

    def forward(self, g):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g

        h = subgraphs[0].srcdata["feat"]

        if self.one_hot_encoder is not None:
            x = subgraphs[0].srcdata["x"]
            h = torch.cat([x, h], dim=1)

        h = self.node_encoder(h)
        h = F.relu(h, inplace=True)
        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):

            if self.edge_encoder is not None:
                efeat = subgraphs[i].edata["feat"]
                efeat_emb = self.edge_encoder[i](efeat)
                efeat_emb = F.relu(efeat_emb, inplace=True)
            else:
                efeat_emb = None

            h = self.convs[i](subgraphs[i], h, efeat_emb).flatten(1, -1)

            if h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h
            h = self.norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout(h)


        h2 = self.node_encoder2(subgraphs[0].srcdata["feat"])
        h2= F.relu(h2, inplace=True)
        h2 = self.input_drop(h2)

        h2_last = None

        for i in range(self.n_layers):

            if self.edge_encoder2 is not None:
                efeat = subgraphs[i].edata["feat"]
                efeat_emb2 = self.edge_encoder2[i](efeat)
                efeat_emb2 = F.relu(efeat_emb2, inplace=True)
            else:
                efeat_emb2 = None

            h2 = self.convs2[i](subgraphs[i], h2, efeat_emb2).flatten(1, -1)

            if h2_last is not None:
                h2 += h2_last[: h.shape[0], :]

            h2_last = h2
            h2 = self.norms2[i](h2)
            h2 = self.activation(h2, inplace=True)
            h2 = self.dropout(h2)

        pred = self.pred_linear(torch.cat([h, h2], dim=-1))
        return pred