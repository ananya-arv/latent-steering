import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        pred_len: int = 25,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),          
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len * 2),   #(pred_len, 2)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        h = h_n[-1]                 #(B, hidden_dim)
        return self.to_latent(h)    #(B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)                       #(B, pred_len*2)
        return out.view(-1, self.pred_len, 2)

    def forward(self, x: torch.Tensor):
        z    = self.encode(x)
        pred = self.decode(z)
        return pred, z


if __name__ == "__main__":
    model = LSTMModel()
    x     = torch.randn(8, 15, 4) 
    pred, z = model(x)
    print(f"pred: {pred.shape}  expected (8, 25, 2)")
    print(f"z:    {z.shape}     expected (8, 64)")
    print("lstm forward pass good")