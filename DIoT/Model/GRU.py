import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(self, input_dim=20, output_dim=100, hidden_dim=128, layer_dim=3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softmax(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        r_out, h_n = self.gru(x, None)
        out = self.fc1(r_out)
        return out