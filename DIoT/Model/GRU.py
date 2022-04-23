import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(self, input_dim=20, output_dim=100, hidden_dim=128, layer_dim=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.out = nn.Softmax(dim=0)     # 经过softmax之后，反向传播是否会有误差？

    def forward(self, x):
        r_out, h_n = self.gru(x, None)
        fc_out = self.fc(r_out[:, -1, :])     # only use the last step data
        out = self.out(fc_out)
        # print(out.shape)
        return out