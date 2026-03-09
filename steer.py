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
    full = np.concatenate([pred_xy, vxy], axis=1)
    return compute_risk_score(full)


def run_experiment(
    model,
    model_name: str,
    val_loader,
    alphas: np.ndarray = None,
    n_samples: int     = 300,
    device: str        = "cuda",
):
    """full probing + steering experiment for one model"""

    if alphas is None:
        alphas = np.linspace(0.0, 1.5, 16)

    print(f"  {model_name}: Latent Probing + Steering")

    latents, risks = extract_latents(model, val_loader, device)
    print(f"  Latents shape: {latents.shape}")
    print(f"  Latent range:  [{latents.min():.3f}, {latents.max():.3f}]")
    print(f"  Risk   range:  [{risks.min():.3f}, {risks.max():.3f}]")
    w, r2, pca = find_steering_vector(latents, risks, percentile=10)
    obs_list = []
    for obs, _, _ in val_loader:
        for i in range(obs.shape[0]):
            obs_list.append(obs[i : i + 1])
            if len(obs_list) >= n_samples:
                break
        if len(obs_list) >= n_samples:
            break

    baseline_risks = []
    for obs_i in obs_list[:100]:
        pred_u, _, _ = steer_and_decode(model, obs_i, w, 0.0, device)
        baseline_risks.append(risk_from_pred(pred_u))
    baseline_risk = float(np.mean(baseline_risks))
    print(f"\n  Baseline risk (α=0): {baseline_risk:.3f}")
    print(f"  Sweeping α ∈ [0, 1.5] over {len(obs_list)} samples...")

    records = []
    for alpha in alphas:
        risks_s, plaus_s, ade_shifts = [], [], []

        for obs_i in obs_list:
            pred_s, _, _ = steer_and_decode(model, obs_i, w, alpha,  device)
            pred_u, _, _ = steer_and_decode(model, obs_i, w, 0.0,    device)

            risks_s.append(risk_from_pred(pred_s))
            plaus_s.append(float(is_plausible(pred_s)))
            ade_shifts.append(float(np.sqrt(((pred_s - pred_u)**2).sum(-1)).mean()))

        rec = {
            "alpha":        float(alpha),
            "mean_risk":    float(np.mean(risks_s)),
            "plausibility": float(np.mean(plaus_s)),
            "ade_shift":    float(np.mean(ade_shifts)),
            "risk_increase": float(np.mean(risks_s)) - baseline_risk,
        }
        records.append(rec)
        print(
            f"  α={alpha:+5.2f} | Risk={rec['mean_risk']:.3f} "
            f"(+{rec['risk_increase']:.3f}) | "
            f"Plausible={rec['plausibility']:.1%} | "
            f"ADE-shift={rec['ade_shift']:.4f} m"
        )

    plausible_recs = [r for r in records if r["plausibility"] >= 0.50]
    if plausible_recs:
        opt = max(plausible_recs, key=lambda r: r["mean_risk"])
        print(f"\n  optimal spot: α={opt['alpha']:.2f} | "
              f"Risk={opt['mean_risk']:.3f} | "
              f"Plausible={opt['plausibility']:.1%}")
    else:
        opt = records[0]
        print(f"\n no alpha achieves >50% plausibility")

    print(f"\n  Random baseline (α={opt['alpha']:.2f}, 100 samples)...")
    rand_risks, rand_plaus = [], []
    for obs_i in obs_list[:100]:
        w_rand = np.random.randn(*w.shape)
        w_rand = w_rand / (np.linalg.norm(w_rand) + 1e-8)
        pred_r, _, _ = steer_and_decode(model, obs_i, w_rand, opt['alpha'], device)
        rand_risks.append(risk_from_pred(pred_r))
        rand_plaus.append(float(is_plausible(pred_r)))

    rand_risk  = float(np.mean(rand_risks))
    rand_plaus = float(np.mean(rand_plaus))
    delta      = opt['mean_risk'] - rand_risk

    print(f"  Structured risk:  {opt['mean_risk']:.3f} | Plausible: {opt['plausibility']:.1%}")
    print(f"  Random risk:      {rand_risk:.3f}      | Plausible: {rand_plaus:.1%}")
    print(f"  → {'Structured > Random by ' + f'{delta:.3f}' if delta > 0.01 else 'no clear advantage'}")

    return records, r2, rand_risk, rand_plaus, opt, w


def plot_results(lstm_records, tf_records, lstm_r2, tf_r2,
                 lstm_sweet, tf_sweet):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Latent Space Steering: LSTM vs Transformer", fontsize=13, fontweight="bold")

    al = [r["alpha"] for r in lstm_records]
    at = [r["alpha"] for r in tf_records]

    axes[0].plot(al, [r["mean_risk"] for r in lstm_records], "b-o", ms=4, label="LSTM")
    axes[0].plot(at, [r["mean_risk"] for r in tf_records],   "r-s", ms=4, label="Transformer")
    axes[0].axvline(lstm_sweet["alpha"], color="blue",  ls="--", alpha=0.4, label=f"LSTM sweet spot")
    axes[0].axvline(tf_sweet["alpha"],   color="red",   ls="--", alpha=0.4, label=f"TF sweet spot")
    axes[0].set_xlabel("Steering α"); axes[0].set_ylabel("Mean Risk Score")
    axes[0].set_title("Risk vs Steering Magnitude")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    axes[1].plot(al, [r["plausibility"] for r in lstm_records], "b-o", ms=4, label="LSTM")
    axes[1].plot(at, [r["plausibility"] for r in tf_records],   "r-s", ms=4, label="Transformer")
    axes[1].axhline(0.5, color="gray", ls=":", alpha=0.7, label="50% threshold")
    axes[1].set_xlabel("Steering α"); axes[1].set_ylabel("Plausibility Rate")
    axes[1].set_title("Physical Plausibility vs Steering")
    axes[1].set_ylim(0, 1.05); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    bars = axes[2].bar(["LSTM", "Transformer"], [lstm_r2, tf_r2],
                        color=["steelblue", "tomato"], width=0.4,
                        edgecolor="k", linewidth=0.7)
    axes[2].set_ylabel("Linear R² (risk ~ z)")
    axes[2].set_title("Latent Space Linearity\n(higher = more structured)")
    axes[2].set_ylim(0, 1)
    for bar, v in zip(bars, [lstm_r2, tf_r2]):
        axes[2].text(bar.get_x() + bar.get_width()/2, v + 0.02,
                     f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = "results/main_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


if __name__ == "__main__":
    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/content/SDD/annotations/"
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_loader, _, _ = get_dataloaders(DATA_DIR, batch_size=128, rare_threshold=0.9)

    lstm = LSTMModel()
    lstm.load_state_dict(torch.load("checkpoints/lstm_best_v1.pt", map_location="cpu"))
    lstm = lstm.to(DEVICE)

    transformer = TransformerModel()
    transformer.load_state_dict(torch.load("checkpoints/transformer_best_v1.pt", map_location="cpu"))
    transformer = transformer.to(DEVICE)

    lstm_rec, lstm_r2, lstm_rand, lstm_rand_plaus, lstm_opt, lstm_w = run_experiment(
        lstm, "LSTM", val_loader, device=DEVICE)

    tf_rec, tf_r2, tf_rand, tf_rand_plaus, tf_opt, tf_w = run_experiment(
        transformer, "Transformer", val_loader, device=DEVICE)

    plot_results(lstm_rec, tf_rec, lstm_r2, tf_r2, lstm_opt, tf_opt)

    table = (
        "  SUMMARY TABLE (for paper)\n"
        f"{'Metric':<40} {'LSTM':>10} {'Transformer':>12}\n"
        + "-"*65 + "\n"
        f"{'Val ADE (m)':<40} {'0.0713':>10} {'0.0576':>12}\n"
        f"{'Val FDE (m)':<40} {'0.1339':>10} {'0.1057':>12}\n"
        f"{'Linear R² (latent linearity)':<40} {lstm_r2:>10.4f} {tf_r2:>12.4f}\n"
        f"{'Optimal spot α':<40} {lstm_opt['alpha']:>10.2f} {tf_opt['alpha']:>12.2f}\n"
        f"{'Structured risk at optimal spot':<40} {lstm_opt['mean_risk']:>10.4f} {tf_opt['mean_risk']:>12.4f}\n"
        f"{'Random baseline risk':<40} {lstm_rand:>10.4f} {tf_rand:>12.4f}\n"
        f"{'Structured advantage':<40} {lstm_opt['mean_risk']-lstm_rand:>10.4f} {tf_opt['mean_risk']-tf_rand:>12.4f}\n"
        f"{'Plausibility at optimal spot':<40} {lstm_opt['plausibility']:>10.4f} {tf_opt['plausibility']:>12.4f}\n"
    )
    print(table)
    with open("results/summary_table.txt", "w") as f:
        f.write(table)
    print("Saved → results/summary_table.txt")
    print("\nNext: python visualize.py /content/SDD/annotations/")















