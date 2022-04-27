class GATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(dataset.num_features, hidden_channels,
                                  heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(heads * hidden_channels, hidden_channels, heads))
        self.convs.append(
            GATv2Conv(heads * hidden_channels, out_channels, heads, concat=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(Linear(dataset.num_features, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.skips.append(
                Linear(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(Linear(hidden_channels * heads, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.25, training=self.training)

            x = x + self.skips[i](x_target)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in test_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.elu(x)
                
                x = x + self.skips[i](x_target)
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all