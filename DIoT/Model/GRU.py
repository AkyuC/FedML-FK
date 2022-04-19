import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, k_lookback):
        """
            layer_dim == 3
            k_lookback == 20
        """
        super().__init__()
        self.gru_list = list()
        self.k_lookback = k_lookback
        for i in range(k_lookback):
            self.gru_list.append(
                    nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
                )

    def forward(self, x):
        output = list()     # transition ?
        r_out, h_n = self.gru(x[0], None)
        output.append(r_out)
        for i in range(self.k_lookback-1):
            r_out, h_n = self.gru(x[i+1], h_n)
            output.append(r_out)
        return output