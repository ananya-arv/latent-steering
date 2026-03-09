import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   #(1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        latent_dim: int = 64,
        pred_len: int = 25,
        dropout: float = 0.2,
        dim_feedforward: int = 256,
    ):
        super().__init__()
        self.pred_len  = pred_len
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.to_latent = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len * 2),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)      #(B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)     #(B, T, d_model)
        x = x.mean(dim=1)           #(B, d_model)
        return self.to_latent(x)    #(B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, self.pred_len, 2)

    def forward(self, x: torch.Tensor):
        z    = self.encode(x)
        pred = self.decode(z)
        return pred, z


if __name__ == "__main__":
    model   = TransformerModel()
    x       = torch.randn(8, 15, 4)
    pred, z = model(x)
    print(f"pred: {pred.shape} ")
    print(f"z:    {z.shape} ")
    print("TransformerModel forward pass good")