import torch
import torch.nn as nn

def vectorized_graph_pooling(x, graph_id, mode='sum', total_graphs=None):
    if total_graphs is None:
        total_graphs = int(graph_id.max().item() + 1) if graph_id.numel() > 0 else 0
    out = torch.zeros(total_graphs, x.size(1), device=x.device, dtype=x.dtype)
    if x.size(0) > 0:
        index = graph_id.unsqueeze(1).expand_as(x)
        out.scatter_add_(0, index, x)
        if mode == 'mean':
            counts = torch.bincount(graph_id, minlength=total_graphs).float().unsqueeze(1)
            out = out / (counts + 1e-6)
    return out

class CSC(nn.Module):
    def __init__(self, cin, cout, m, k_node, stride=1, use_act=True, use_norm=True, dropout=0.0):
        super().__init__()
        self.m = m
        self.k_node = k_node
        self.stride = stride
        self.use_norm = use_norm
        self.weight = nn.Parameter(torch.empty(cout, cin, k_node, 1 + m))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.bn1 = nn.BatchNorm1d(cout) if self.use_norm else nn.Identity()
        self.act = nn.ReLU() if use_act else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.bias)

    def forward(self, x, graph_id):
        device = x.device
        if x.size(0) == 0:
            return x, graph_id
        unique_g = torch.unique(graph_id)
        outputs = []
        out_ids = []
        for g in unique_g.tolist():
            idx = (graph_id == g).nonzero(as_tuple=False).squeeze(1)
            xg = x[idx]
            n_g = xg.size(0)
            if n_g < self.k_node:
                pad_size = self.k_node - n_g
                pad_shape = list(xg.shape)
                pad_shape[0] = pad_size
                padding = torch.zeros(pad_shape, device=device, dtype=xg.dtype)
                xg = torch.cat([xg, padding], dim=0)
            xg_win = xg.unfold(0, self.k_node, self.stride).permute(0, 4, 1, 2, 3)
            y_g = torch.einsum("ockr,wkcpr->wpo", self.weight, xg_win)
            y_g = y_g + self.bias.view(1, 1, -1)
            outputs.append(y_g)
            n_out_g = y_g.shape[0]
            out_ids.append(torch.full((n_out_g,), g, dtype=torch.long, device=device))
        if not outputs:
            return x.new_zeros((0, self.weight.size(0), x.size(2))), graph_id.new_zeros((0,))
        y = torch.cat(outputs, dim=0)
        new_graph_id = torch.cat(out_ids, dim=0)
        y = y.permute(0, 2, 1)
        if self.use_norm:
            y = self.bn1(y)
        y = self.act(y)
        y = self.dropout(y)
        return y, new_graph_id

class CSCN(nn.Module):
    def __init__(self, num_classes, feat_dim, layers, m, channels, k_nodes, strides, fc_dim=64, dropout=0.5, readout_mode='sum'):
        super().__init__()
        self.m = m
        self.readout_mode = readout_mode
        self.convs = nn.ModuleList()
        curr_c = feat_dim
        current_P = (m + 1) ** (layers - 1)
        classifier_in_dim = 0
        for i in range(len(channels)):
            out_c = channels[i]
            k_n = k_nodes[i]
            s_n = strides[i]
            conv = CSC(curr_c, out_c, m, k_node=k_n, stride=s_n, use_norm=True, dropout=dropout)
            self.convs.append(conv)
            curr_c = out_c
            current_P = current_P // (m + 1)
            layer_dim = out_c * current_P
            classifier_in_dim += layer_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes)
        )

    def forward(self, x, graph_id):
        total_graphs = int(graph_id.max().item() + 1) if graph_id.numel() > 0 else 0
        layer_embeddings = []
        for i, conv in enumerate(self.convs):
            x, graph_id = conv(x, graph_id)
            if x.size(0) == 0:
                break
            N, C, P = x.shape
            curr_x_flat = x.reshape(N, -1)
            graph_feat = vectorized_graph_pooling(curr_x_flat, graph_id, mode=self.readout_mode, total_graphs=total_graphs)
            layer_embeddings.append(graph_feat)
            if i < len(self.convs) - 1:
                R = self.m + 1
                if P >= R and P % R == 0:
                    x = x.reshape(N, C, P // R, R)
                else:
                    raise ValueError(f"Dimension mismatch layer {i}")
        if len(layer_embeddings) > 0:
            out = torch.cat(layer_embeddings, dim=1)
        else:
            out = torch.zeros(total_graphs, self.classifier[0].in_features, device=x.device)
        out = self.classifier(out)
        return out