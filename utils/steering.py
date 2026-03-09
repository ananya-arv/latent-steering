"""
Latent space probing and steering utilities.
--> extract the latent vectors z for all the training samples
--> fit linear probe: risk ~ z  → shows us if risk is linearly encoded
--> find steering direction w in latent space that points toward danger
--> at inference: z' = z + alpha * w, then decode z' to get a steered trajectory

2 methods for finding w:
--> mean diff:  w = mean(z | high-risk) - mean(z | low-risk)
--> linear probe:     w = weights of Ridge regression risk ~ z        
    ~~ should point in similar directions if risk is linearly encoded
"""

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def extract_latents(model, dataloader, device: str = "cuda"):
    model.eval()
    all_z, all_r = [], []

    with torch.no_grad():
        for obs, _, risk in dataloader:
            z = model.encode(obs.to(device)).cpu().numpy()
            all_z.append(z)
            all_r.append(risk.numpy())

    return np.vstack(all_z), np.concatenate(all_r)


def find_steering_vector(latents: np.ndarray, risks: np.ndarray, percentile: int = 10):
    """
    find steering direction w using PCA-whitened mean difference.
    --> PCA whiten the latents (zero mean, identity covariance)
    --> compute mean diff in whitened space: w = mean(z_hi) - mean(z_lo)
    --> normalize w to unit length
    --> auto-correct direction: ensure alpha > 0 → higher risk
    --> fit Ridge probe for R2 diagnostic

    """
    pca = PCA(n_components=latents.shape[1], whiten=True)
    z_white = pca.fit_transform(latents)   #(N, latent_dim)
    lo_thresh = np.percentile(risks, percentile)
    hi_thresh = np.percentile(risks, 100 - percentile)
    lo_mask   = risks <= lo_thresh
    hi_mask   = risks >= hi_thresh

    w_white = z_white[hi_mask].mean(0) - z_white[lo_mask].mean(0)
    w_white = w_white / (np.linalg.norm(w_white) + 1e-8)
    w_raw = pca.inverse_transform(w_white) - pca.inverse_transform(np.zeros_like(w_white))
    w_raw = w_raw / (np.linalg.norm(w_raw) + 1e-8)

    lo_risk = float(risks[lo_mask].mean())
    hi_risk = float(risks[hi_mask].mean())
    risk_corr = np.corrcoef(latents @ w_raw, risks)[0, 1]
    if risk_corr < 0:
        w_raw = -w_raw
        print("  Auto-corrected w direction (risk_corr was negative)")

    scaler   = StandardScaler()
    z_scaled = scaler.fit_transform(latents)
    probe    = Ridge(alpha=1.0)
    probe.fit(z_scaled, risks)
    r2 = probe.score(z_scaled, risks)

    print(f"  Linear R²:             {r2:.4f}  "
          f"({'good linear structure' if r2 > 0.25 else 'weak linear structure'})")
    print(f"  Risk corr with w:      {risk_corr:.4f}  "
          f"({'aligned' if abs(risk_corr) > 0.1 else 'weak alignment'})")
    print(f"  Low-risk  mean:        {lo_risk:.3f}  (bottom {percentile}%)")
    print(f"  High-risk mean:        {hi_risk:.3f}  (top {percentile}%)")
    print(f"  Risk gap:              {hi_risk - lo_risk:.3f}")
    print(f"  High-risk samples:     {hi_mask.sum()}")
    print(f"  Low-risk  samples:     {lo_mask.sum()}")

    return w_raw, r2, pca


def steer_and_decode(model, obs: torch.Tensor, w: np.ndarray,
                     alpha: float, device: str = "cuda",
                     z_min: float = -0.99, z_max: float = 0.99):
    """
    gen a trajectory steered alpha steps along w.
    """
    model.eval()
    with torch.no_grad():
        z = model.encode(obs.to(device)).cpu().numpy()   #(1, latent_dim)

        z_steered = z + alpha * w
        z_steered = np.clip(z_steered, z_min, z_max) 

        z_t  = torch.tensor(z_steered, dtype=torch.float32).to(device)
        pred = model.decode(z_t).cpu().numpy()[0]      

    return pred, z[0], z_steered[0]



def is_plausible(traj_xy: np.ndarray,
                 dt: float = 1.0 / 30,
                 max_speed: float = 3.5) -> bool:
    if len(traj_xy) < 2:
        return True

    step_dists = np.sqrt(((np.diff(traj_xy, axis=0))**2).sum(-1))
    speeds     = step_dists / dt

    return bool(speeds.max() <= max_speed)