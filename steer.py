"""
steering experiment: probe latent space, sweep alpha, collect all metrics

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.dataset import get_dataloaders, compute_risk_score, normalize_window
from utils.steering import (
    extract_latents, find_steering_vector,
    steer_and_decode, is_plausible,
)
from models.lstm import LSTMModel
from models.transformer import TransformerModel

os.makedirs("results", exist_ok=True)
DT = 1.0 / 30


def risk_from_pred(pred_xy: np.ndarray) -> float:
    vxy  = np.diff(pred_xy, axis=0, prepend=pred_xy[:1]) / DT
    full = np.concatenate([pred_xy, vxy], axis=1)   #(T, 4)
    return compute_risk_score(full)


def run_experiment(
    model,
    model_name: str,
    val_loader,
    alphas: np.ndarray = None,
    n_samples: int = 300,
    device: str = "cuda",
):
    """
    full probing + steering experiment for one model.
    --> extract all latents from val_loader
    --> fit steering vector (mean-diff + linear probe)
    --> for each alpha: steer N sample trajectories, compute metrics
    --> random-direction baseline at alpha=1.0
    """
    if alphas is None:
        alphas = np.linspace(-2.0, 2.0, 17)

    print(f"\n{'='*55}")
    print(f"  {model_name}: Latent Probing + Steering")
    print(f"{'='*55}")

    latents, risks = extract_latents(model, val_loader, device)
    print(f"  Latents shape: {latents.shape}")
    w_mean, w_probe, r2, scaler = find_steering_vector(latents, risks)
    obs_list = []
    for obs, _, _ in val_loader:
        for i in range(obs.shape[0]):
            obs_list.append(obs[i : i + 1])
            if len(obs_list) >= n_samples:
                break
        if len(obs_list) >= n_samples:
            break
    print(f"\n  Sweeping alpha over {len(obs_list)} samples...")

    records = []
    for alpha in alphas:
        risks_s, plaus_s, ade_shifts = [], [], []

        for obs_i in obs_list:
            pred_s, _, _ = steer_and_decode(model, obs_i, w_probe, alpha, scaler, device)
            pred_u, _, _ = steer_and_decode(model, obs_i, w_probe, 0.0,   scaler, device)

            risks_s.append(risk_from_pred(pred_s))
            plaus_s.append(float(is_plausible(pred_s)))
            ade_shifts.append(float(np.sqrt(((pred_s - pred_u) ** 2).sum(-1)).mean()))

        rec = {
            "alpha":        float(alpha),
            "mean_risk":    float(np.mean(risks_s)),
            "plausibility": float(np.mean(plaus_s)),
            "ade_shift":    float(np.mean(ade_shifts)),
        }
        records.append(rec)
        print(
            f"  α={alpha:+5.2f} | Risk={rec['mean_risk']:.3f} | "
            f"Plausible={rec['plausibility']:.1%} | ADE-shift={rec['ade_shift']:.3f} m"
        )

    print(f"\n  Random baseline (α=1.0, 100 samples)...")
    rand_risks = []
    for obs_i in obs_list[:100]:
        w_rand = np.random.randn(*w_probe.shape)
        w_rand = w_rand / (np.linalg.norm(w_rand) + 1e-8)
        pred_r, _, _ = steer_and_decode(model, obs_i, w_rand, 1.0, scaler, device)
        rand_risks.append(risk_from_pred(pred_r))

    rand_risk = float(np.mean(rand_risks))
    alpha1_rec = min(records, key=lambda r: abs(r["alpha"] - 1.0))

    print(f"\n  Structured (α=1.0) risk: {alpha1_rec['mean_risk']:.3f}")
    print(f"  Random     (α=1.0) risk: {rand_risk:.3f}")
    delta = alpha1_rec["mean_risk"] - rand_risk
    print(f"  → {'Structured > Random by ' + f'{delta:.3f}' if delta > 0.02 else 'no clear advantage over random'}")

    return records, r2, rand_risk, w_probe, scaler


def plot_results(lstm_records, tf_records, lstm_r2, tf_r2):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Latent Space Steering: LSTM vs Transformer", fontsize=13, fontweight="bold")

    al = [r["alpha"] for r in lstm_records]
    at = [r["alpha"] for r in tf_records]

    # Panel 1: Risk vs alpha
    axes[0].plot(al, [r["mean_risk"] for r in lstm_records], "b-o", ms=4, label="LSTM")
    axes[0].plot(at, [r["mean_risk"] for r in tf_records],   "r-s", ms=4, label="Transformer")
    axes[0].axvline(0, color="gray", ls=":", alpha=0.5)
    axes[0].set_xlabel("Steering α"); axes[0].set_ylabel("Mean Risk Score")
    axes[0].set_title("Risk vs Steering Magnitude")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Panel 2: Plausibility vs alpha
    axes[1].plot(al, [r["plausibility"] for r in lstm_records], "b-o", ms=4, label="LSTM")
    axes[1].plot(at, [r["plausibility"] for r in tf_records],   "r-s", ms=4, label="Transformer")
    axes[1].axvline(0, color="gray", ls=":", alpha=0.5)
    axes[1].set_xlabel("Steering α"); axes[1].set_ylabel("Plausibility Rate")
    axes[1].set_title("Physical Plausibility vs Steering")
    axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(alpha=0.3)

    # Panel 3: R² bar chart
    bars = axes[2].bar(["LSTM", "Transformer"], [lstm_r2, tf_r2],
                        color=["steelblue", "tomato"], width=0.4, edgecolor="k", linewidth=0.7)
    axes[2].set_ylabel("Linear R² (risk ~ z)")
    axes[2].set_title("Latent Space Linearity\n(higher = more structured)")
    axes[2].set_ylim(0, 1)
    for bar, v in zip(bars, [lstm_r2, tf_r2]):
        axes[2].text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                     f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = "results/main_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


if __name__ == "__main__":
    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "OpenTraj/datasets/SDD/"
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_loader, test_loader, rare_loader = get_dataloaders(DATA_DIR, batch_size=128)

    lstm = LSTMModel()
    lstm.load_state_dict(torch.load("checkpoints/lstm_best_v2.pt", map_location="cpu"))
    lstm = lstm.to(DEVICE)

    transformer = TransformerModel()
    transformer.load_state_dict(torch.load("checkpoints/transformer_best_v2.pt", map_location="cpu"))
    transformer = transformer.to(DEVICE)

    lstm_rec, lstm_r2, lstm_rand, lstm_w, lstm_sc = run_experiment(
        lstm, "LSTM", val_loader, device=DEVICE)

    tf_rec, tf_r2, tf_rand, tf_w, tf_sc = run_experiment(
        transformer, "Transformer", val_loader, device=DEVICE)

    plot_results(lstm_rec, tf_rec, lstm_r2, tf_r2)

    alpha1_lstm = min(lstm_rec, key=lambda r: abs(r["alpha"] - 1.0))
    alpha1_tf   = min(tf_rec,   key=lambda r: abs(r["alpha"] - 1.0))

    table = (
        "  SUMMARY TABLE\n"
        f"{'Metric':<38} {'LSTM':>10} {'Transformer':>12}\n"
        + "-"*62 + "\n"
        f"{'Linear R² (latent linearity)':<38} {lstm_r2:>10.4f} {tf_r2:>12.4f}\n"
        f"{'Structured risk  (α=1.0)':<38} {alpha1_lstm['mean_risk']:>10.4f} {alpha1_tf['mean_risk']:>12.4f}\n"
        f"{'Random baseline risk (α=1.0)':<38} {lstm_rand:>10.4f} {tf_rand:>12.4f}\n"
        f"{'Plausibility at α=1.0':<38} {alpha1_lstm['plausibility']:>10.4f} {alpha1_tf['plausibility']:>12.4f}\n"
    )
    print(table)

    with open("results/summary_table.txt", "w") as f:
        f.write(table)
    print("Saved → results/summary_table.txt")
    print("\nNext step: python visualize.py")