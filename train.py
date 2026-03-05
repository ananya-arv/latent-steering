"""
training script

Outputs:
    checkpoints/lstm_best.pt        — best LSTM weights
    checkpoints/transformer_best.pt — best Transformer weights
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils.dataset import get_dataloaders
from models.lstm import LSTMModel
from models.transformer import TransformerModel

os.makedirs("checkpoints", exist_ok=True)


def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    avg displacement error (avg Euclidean distance over all timesteps)
    """
    return torch.sqrt(((pred - target) ** 2).sum(-1)).mean()


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    final displacement error (Euclidean distance at last predicted timestep)
    """
    return torch.sqrt(((pred[:, -1] - target[:, -1]) ** 2).sum(-1)).mean()


def run_epoch(model, loader, optimizer, device, train: bool = True):
    model.train() if train else model.eval()
    losses, ades, fdes = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for obs, pred_gt, _ in tqdm(loader, leave=False, desc="train" if train else "val"):
            obs     = obs.to(device)
            pred_gt = pred_gt[:, :, :2].to(device) 

            pred, _ = model(obs)

            loss = ade(pred, pred_gt)   #train on the avg displacement error
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            losses.append(loss.item())
            ades.append(ade(pred, pred_gt).item())
            fdes.append(fde(pred, pred_gt).item())

    return float(np.mean(losses)), float(np.mean(ades)), float(np.mean(fdes))



def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int   = 60,
    lr: float     = 1e-3,
    device: str   = "cuda",
    save_path: str = None,
    model_name: str = "Model",
):
    """
    optimizer: Adam + CosineAnnealingLR.
    best checkpoint (lowest val ADE)
    """
    model     = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_ade = float("inf")
    best_state   = None
    history      = {"train_ade": [], "val_ade": [], "val_fde": []}

    print(f"  Training {model_name}  ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"  Device: {device}  |  Epochs: {epochs}  |  LR: {lr}")

    for epoch in range(1, epochs + 1):
        train_loss, train_ade_val, _ = run_epoch(
            model, train_loader, optimizer, device, train=True)
        _,          val_ade_val,   val_fde_val = run_epoch(
            model, val_loader,   optimizer, device, train=False)
        scheduler.step()

        history["train_ade"].append(train_ade_val)
        history["val_ade"].append(val_ade_val)
        history["val_fde"].append(val_fde_val)

        if val_ade_val < best_val_ade:
            best_val_ade = val_ade_val
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train ADE: {train_ade_val:.4f} | "
                f"Val ADE: {val_ade_val:.4f} | "
                f"Val FDE: {val_fde_val:.4f}"
                + (" ← best" if val_ade_val == best_val_ade else "")
            )

    model.load_state_dict(best_state)
    if save_path:
        torch.save(best_state, save_path)
        print(f"\n  Saved best model → {save_path}")

    print(f"  Best Val ADE: {best_val_ade:.4f} m")
    return model, history


if __name__ == "__main__":
    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "OpenTraj/datasets/SDD/"
    EPOCHS   = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {DEVICE}")
    print(f"\nLoading data from: {DATA_DIR}")
    train_loader, val_loader, test_loader, rare_loader = get_dataloaders(
        DATA_DIR, batch_size=128
    )

    lstm, lstm_hist = train_model(
        LSTMModel(),
        train_loader, val_loader,
        epochs=EPOCHS, device=DEVICE,
        save_path="checkpoints/lstm_best.pt",
        model_name="LSTM",
    )

    transformer, tf_hist = train_model(
        TransformerModel(),
        train_loader, val_loader,
        epochs=EPOCHS, device=DEVICE,
        save_path="checkpoints/transformer_best.pt",
        model_name="Transformer",
    )

    print("  TRAINING COMPLETE")
    print(f"  LSTM best val ADE:        {min(lstm_hist['val_ade']):.4f} m")
    print(f"  LSTM best val FDE:        {min(lstm_hist['val_fde']):.4f} m")
    print(f"  Transformer best val ADE: {min(tf_hist['val_ade']):.4f} m")
    print(f"  Transformer best val FDE: {min(tf_hist['val_fde']):.4f} m")