"""
Generate all paper figures.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

from utils.dataset import get_dataloaders, compute_risk_score, normalize_window
from utils.steering import extract_latents, find_steering_vector, steer_and_decode
from models.lstm import LSTMModel
from models.transformer import TransformerModel

os.makedirs("results", exist_ok=True)
DT = 1.0 / 30


def risk_from_pred(pred_xy: np.ndarray) -> float:
    vxy  = np.diff(pred_xy, axis=0, prepend=pred_xy[:1]) / DT
    full = np.concatenate([pred_xy, vxy], axis=1)
    return compute_risk_score(full)


def constant_velocity_pred(obs: np.ndarray, pred_len: int = 25) -> np.ndarray:
    """Extrapolate last observed velocity (standard baseline)."""
    last_pos = obs[-1, :2]
    last_vel = obs[-1, 2:]
    return np.array([last_pos + last_vel * DT * (t + 1) for t in range(pred_len)])


def collect_obs(val_loader, n: int = 300):
    obs_list = []
    for obs, _, _ in val_loader:
        for i in range(obs.shape[0]):
            obs_list.append(obs[i : i + 1])
            if len(obs_list) >= n:
                return obs_list
    return obs_list


def fig1_pca_latent(lstm, transformer, val_loader, device):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Latent Space PCA — Colored by Risk Score\n"
                 "(clear green→red gradient = risk linearly encoded)",
                 fontsize=12, fontweight="bold")

    for ax, model, name in zip(axes, [lstm, transformer], ["LSTM", "Transformer"]):
        latents, risks = extract_latents(model, val_loader, device)

        idx = np.random.choice(len(latents), min(8000, len(latents)), replace=False)
        pca = PCA(n_components=2)
        z2  = pca.fit_transform(latents[idx])
        r   = risks[idx]

        var = pca.explained_variance_ratio_
        sc  = ax.scatter(z2[:, 0], z2[:, 1], c=r, cmap="RdYlGn_r",
                         alpha=0.35, s=6, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label="Risk Score")
        ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")

        from numpy.polynomial import polynomial as P
        coeffs = np.polyfit(z2[:, 0], r, 1)
        x_line = np.linspace(z2[:, 0].min(), z2[:, 0].max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line) * 
                (z2[:, 1].max() - z2[:, 1].min()) + z2[:, 1].mean(),
                'k--', lw=1.5, alpha=0.5, label='Risk trend')

        r2 = np.corrcoef(z2[:, 0], r)[0, 1] ** 2
        ax.set_title(f"{name}  (R²={r2:.3f} along PC1)", fontsize=11)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    out = "results/fig1_pca_latent.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")



def fig2_steering_examples(lstm, transformer, val_loader, 
                            w_lstm, w_tf, device, n_examples=4):
    """
    For n_examples trajectories, show observed path + steered predictions
    at α=0.0, 0.5, 1.0, 1.5 for both models side by side.
    Color: blue=observed, green→red gradient = low→high alpha.
    """
    alphas = [0.0, 0.5, 1.0, 1.5]
    colors = ["#2196F3", "#FFC107", "#FF5722", "#B71C1C"]

    obs_list = collect_obs(val_loader, n_examples)

    fig, axes = plt.subplots(n_examples, 2, figsize=(12, 4 * n_examples))
    fig.suptitle("Steered Trajectory Examples\n"
                 "Blue=Observed  |  Blue→Red = α increasing (low→high risk)",
                 fontsize=12, fontweight="bold")

    for row, obs_i in enumerate(obs_list[:n_examples]):
        obs_np = obs_i[0].numpy()

        for col, (model, model_name, w) in enumerate([
            (lstm, "LSTM", w_lstm),
            (transformer, "Transformer", w_tf)
        ]):
            ax = axes[row, col]

            ax.plot(obs_np[:, 0], obs_np[:, 1], "k-o", ms=3,
                    lw=2, label="Observed", zorder=5)
            ax.plot(obs_np[-1, 0], obs_np[-1, 1], "k*", ms=12, zorder=6)

            for alpha, color in zip(alphas, colors):
                pred, _, _ = steer_and_decode(model, obs_i, w, alpha, device)
                risk       = risk_from_pred(pred)
                ax.plot(pred[:, 0], pred[:, 1], "-o", ms=2,
                        color=color, lw=2,
                        label=f"α={alpha:.1f} (r={risk:.2f})")
                ax.plot(pred[-1, 0], pred[-1, 1], "^",
                        color=color, ms=7, zorder=5)

            ax.set_title(f"{model_name} — Example {row+1}", fontsize=10)
            ax.legend(fontsize=7, loc="best")
            ax.set_aspect("equal")
            ax.grid(alpha=0.25)
            ax.set_facecolor("#fafafa")

    plt.tight_layout()
    out = "results/fig2_steering_examples.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")



def fig3_kde_risk(transformer, val_loader, w_tf, device):
    """
    KDE showing how steering shifts the risk distribution rightward.
    Training data = green. Steered at increasing alpha = orange→red.
    """
    obs_list = collect_obs(val_loader, 300)

    real_risks = []
    for _, _, risk in val_loader:
        real_risks.extend(risk.numpy().tolist())
    real_risks = np.array(real_risks)

    alphas_kde = [0.0, 0.5, 1.0, 1.5]
    steered = []
    for alpha in alphas_kde:
        risks_a = []
        for obs_i in obs_list:
            pred, _, _ = steer_and_decode(transformer, obs_i, w_tf, alpha, device)
            risks_a.append(risk_from_pred(pred))
        steered.append((alpha, np.array(risks_a)))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.linspace(0, 1, 300)

    kde_real = gaussian_kde(real_risks, bw_method=0.12)
    ax.fill_between(x, kde_real(x), alpha=0.20, color="forestgreen")
    ax.plot(x, kde_real(x), "g-", lw=2.5, label="Training data")

    reds = plt.cm.YlOrRd(np.linspace(0.3, 0.95, len(alphas_kde)))
    for (alpha, risks_a), color in zip(steered, reds):
        kde = gaussian_kde(risks_a, bw_method=0.12)
        ax.plot(x, kde(x), "-", color=color, lw=2.0,
                label=f"Steered α={alpha:.1f} (mean={risks_a.mean():.3f})")

    ax.axvline(0.85, color="gray", ls="--", lw=1.2, alpha=0.7,
               label="Rare threshold (0.85)")
    ax.set_xlabel("Risk Score", fontsize=12)
    ax.set_ylabel("Density",    fontsize=12)
    ax.set_title("Risk Distribution Shift via Latent Steering (Transformer)\n"
                 "Steering pushes mass into the high-risk tail",
                 fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 1); ax.grid(alpha=0.2)
    plt.tight_layout()

    out = "results/fig3_kde_risk.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")



def fig4_cv_baseline(lstm, transformer, test_loader, device):
    """
    Compare LSTM, Transformer, and constant-velocity baseline on test set.
    Reports test ADE and FDE
    """
    def eval_model(model, loader):
        model.eval()
        ades, fdes = [], []
        with torch.no_grad():
            for obs, pred_gt, _ in loader:
                pred, _  = model(obs.to(device))
                gt_xy    = pred_gt[:, :, :2].to(device)
                ade = torch.sqrt(((pred - gt_xy)**2).sum(-1)).mean(dim=1)
                fde = torch.sqrt(((pred[:, -1] - gt_xy[:, -1])**2).sum(-1))
                ades.extend(ade.cpu().numpy())
                fdes.extend(fde.cpu().numpy())
        return np.mean(ades), np.mean(fdes)

    def eval_cv(loader):
        ades, fdes = [], []
        for obs, pred_gt, _ in loader:
            obs_np    = obs.numpy()
            pred_np   = pred_gt[:, :, :2].numpy()
            for b in range(obs_np.shape[0]):
                cv_pred = constant_velocity_pred(obs_np[b], pred_len=pred_np.shape[1])
                diff    = cv_pred - pred_np[b]
                ade     = np.sqrt((diff**2).sum(-1)).mean()
                fde     = np.sqrt((diff[-1]**2).sum())
                ades.append(ade); fdes.append(fde)
        return np.mean(ades), np.mean(fdes)

    print("  Evaluating on test set...")
    lstm_ade,  lstm_fde  = eval_model(lstm,        test_loader)
    tf_ade,    tf_fde    = eval_model(transformer, test_loader)
    cv_ade,    cv_fde    = eval_cv(test_loader)

    print(f"  Constant Velocity — ADE: {cv_ade:.4f}  FDE: {cv_fde:.4f}")
    print(f"  LSTM              — ADE: {lstm_ade:.4f}  FDE: {lstm_fde:.4f}")
    print(f"  Transformer       — ADE: {tf_ade:.4f}  FDE: {tf_fde:.4f}")

    models  = ["Const.\nVelocity", "LSTM", "Transformer"]
    ade_vals = [cv_ade, lstm_ade, tf_ade]
    fde_vals = [cv_fde, lstm_fde, tf_fde]
    colors   = ["#9E9E9E", "#2196F3", "#F44336"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Test Set Evaluation: ADE and FDE (meters)\n"
                 "Lower is better", fontsize=12, fontweight="bold")

    for ax, vals, metric in zip(axes, [ade_vals, fde_vals], ["ADE (m)", "FDE (m)"]):
        bars = ax.bar(models, vals, color=colors, edgecolor="k",
                      linewidth=0.7, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=11)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = "results/fig4_cv_baseline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")

    return {
        "cv":  {"ade": cv_ade,   "fde": cv_fde},
        "lstm": {"ade": lstm_ade, "fde": lstm_fde},
        "tf":   {"ade": tf_ade,   "fde": tf_fde},
    }



def fig5_summary(lstm_r2, tf_r2, lstm_adv, tf_adv):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Summary: Latent Space Quality Metrics",
                 fontsize=12, fontweight="bold")

    axes[0].bar(["LSTM", "Transformer"], [lstm_r2, tf_r2],
                color=["#2196F3", "#F44336"], edgecolor="k",
                linewidth=0.7, width=0.45)
    axes[0].set_ylabel("Linear R² (risk ~ latent)", fontsize=11)
    axes[0].set_title("Latent Space Linearity\n(higher = more structured)", fontsize=10)
    axes[0].set_ylim(0, 1)
    for i, v in enumerate([lstm_r2, tf_r2]):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center",
                     fontsize=13, fontweight="bold")
    axes[0].axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6,
                    label="R²=0.5 threshold")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(["LSTM", "Transformer"], [lstm_adv, tf_adv],
                color=["#2196F3", "#F44336"], edgecolor="k",
                linewidth=0.7, width=0.45)
    axes[1].set_ylabel("Structured − Random risk", fontsize=11)
    axes[1].set_title("Steering Advantage over Random\n(higher = more controllable)",
                      fontsize=10)
    axes[1].set_ylim(0, max(lstm_adv, tf_adv) * 1.4)
    for i, v in enumerate([lstm_adv, tf_adv]):
        axes[1].text(i, v + 0.003, f"+{v:.3f}", ha="center",
                     fontsize=13, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = "results/fig5_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")



if __name__ == "__main__":
    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/content/latent-steering/SDD/annotations/"
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {DEVICE}")
    print(f"Loading data from: {DATA_DIR}\n")

    _, val_loader, test_loader, _ = get_dataloaders(
        DATA_DIR, batch_size=128, rare_threshold=0.85
    )

    lstm = LSTMModel()
    lstm.load_state_dict(torch.load("checkpoints/lstm_best.pt", map_location="cpu"))
    lstm = lstm.to(DEVICE)

    transformer = TransformerModel()
    transformer.load_state_dict(torch.load("checkpoints/transformer_best.pt", map_location="cpu"))
    transformer = transformer.to(DEVICE)

    print("Extracting latents and steering vectors...")
    latents_l, risks_l = extract_latents(lstm,        val_loader, DEVICE)
    latents_t, risks_t = extract_latents(transformer, val_loader, DEVICE)
    w_lstm, r2_lstm, _ = find_steering_vector(latents_l, risks_l)
    w_tf,   r2_tf,   _ = find_steering_vector(latents_t, risks_t)
    print(f"  LSTM R²: {r2_lstm:.4f}  |  Transformer R²: {r2_tf:.4f}")

    print("\n[Fig 1] PCA latent space...")
    fig1_pca_latent(lstm, transformer, val_loader, DEVICE)

    print("\n[Fig 2] Steering trajectory examples...")
    fig2_steering_examples(lstm, transformer, val_loader,
                           w_lstm, w_tf, DEVICE, n_examples=4)

    print("\n[Fig 3] KDE risk distribution shift...")
    fig3_kde_risk(transformer, val_loader, w_tf, DEVICE)

    print("\n[Fig 4] Constant-velocity baseline comparison...")
    test_results = fig4_cv_baseline(lstm, transformer, test_loader, DEVICE)

    lstm_adv = 0.141   # hardcoded from steer.py run
    tf_adv   = 0.159

    print("\n[Fig 5] Summary figure...")
    fig5_summary(r2_lstm, r2_tf, lstm_adv, tf_adv)

    print("\n All figures saved to results/")
    print("\n  FINAL TEST RESULTS:")
    print(f"  Constant Velocity ADE: {test_results['cv']['ade']:.4f}m  FDE: {test_results['cv']['fde']:.4f}m")
    print(f"  LSTM              ADE: {test_results['lstm']['ade']:.4f}m  FDE: {test_results['lstm']['fde']:.4f}m")
    print(f"  Transformer       ADE: {test_results['tf']['ade']:.4f}m   FDE: {test_results['tf']['fde']:.4f}m")