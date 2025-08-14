import torch
import torch.nn as nn

class RNNDecoder(nn.Module):
    def __init__(self, in_ch=4, hid=128, out_classes=5):
        super().__init__()
        self.gru = nn.GRU(input_size=in_ch, hidden_size=hid, batch_first=True, num_layers=2, bidirectional=True)
        self.head = nn.Sequential(
            nn.LayerNorm(2*hid),
            nn.ReLU(),
            nn.Linear(2*hid, out_classes)
        )

    def forward(self, x):          # x: (B, T, C)
        out, _ = self.gru(x)
        out = out[:, -1, :]        # last timestep
        return self.head(out)

class TransformerDecoder(nn.Module):
    def __init__(self, in_ch=4, d_model=128, nhead=4, nlayers=3, out_classes=5):
        super().__init__()
        self.embed = nn.Linear(in_ch, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.cls = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_classes)
        )

    def forward(self, x):          # x: (B, T, C)
        z = self.embed(x)
        z = self.enc(z)
        z = z.mean(dim=1)          # mean pooling over time
        return self.cls(z)
