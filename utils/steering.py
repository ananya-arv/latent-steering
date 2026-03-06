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



def extract_latents(model, dataloader, device: str = "cuda"):
    model.eval()
    all_z, all_r = [], []

    with torch.no_grad():
        for obs, _, risk in dataloader:
            z = model.encode(obs.to(device)).cpu().numpy()
            all_z.append(z)
            all_r.append(risk.numpy())

    return np.vstack(all_z), np.concatenate(all_r)



def find_steering_vector(latents: np.ndarray, risks: np.ndarray, percentile: int = 20):
    """
    find steering direction w in raw latent space.
    """
    lo_thresh = np.percentile(risks, percentile)
    hi_thresh = np.percentile(risks, 100 - percentile)
    lo_mask   = risks <= lo_thresh
    hi_mask   = risks >= hi_thresh

    w = latents[hi_mask].mean(0) - latents[lo_mask].mean(0)
    w = w / (np.linalg.norm(w) + 1e-8)

    scaler   = StandardScaler()
    z_scaled = scaler.fit_transform(latents)
    probe    = Ridge(alpha=1.0)
    probe.fit(z_scaled, risks)
    r2 = probe.score(z_scaled, risks)

    print(f"  Linear R²:             {r2:.4f}  "
          f"({'good linear structure' if r2 > 0.25 else 'weak linear structure'})")
    print(f"  Latent mean:           {latents.mean():.4f}  std: {latents.std():.4f}")
    print(f"  Steering vector norm:  {np.linalg.norm(w):.4f} (should be 1.0)")
    print(f"  High-risk samples:     {hi_mask.sum()}")
    print(f"  Low-risk  samples:     {lo_mask.sum()}")

    return w, r2, scaler


def steer_and_decode(model, obs: torch.Tensor, w: np.ndarray,
                     alpha: float, device: str = "cuda"):
    """
    gen a trajectory steered alpha steps along w in RAW latent space.

    process:
        z    = model.encode(obs)          # raw latent
        z'   = z + alpha * w              # steer directly
        pred = model.decode(z')           # decode steered latent

    alpha > 0 → steers toward high-risk behavior
    alpha < 0 → steers toward low-risk  behavior
    alpha = 0 → original unsteered prediction

    """
    model.eval()
    with torch.no_grad():
        z = model.encode(obs.to(device)).cpu().numpy()    #(1, latent_dim)

        z_steered = z + alpha * w                         
        z_t  = torch.tensor(z_steered, dtype=torch.float32).to(device)
        pred = model.decode(z_t).cpu().numpy()[0]         #(pred_len, 2)

    return pred, z[0], z_steered[0]


def is_plausible(traj_xy: np.ndarray,
                 dt: float      = 1.0 / 30,
                 max_speed: float = 6.0,
                 max_accel: float = 4.0) -> bool:
    """
    check whether a generated trajectory satisfies pedestrian physics.
    """
    if len(traj_xy) < 2:
        return True

    velocities = np.diff(traj_xy, axis=0) / dt
    speeds     = np.sqrt((velocities ** 2).sum(-1))
    accels     = np.abs(np.diff(speeds)) / dt

    if speeds.max() > max_speed:
        return False
    if len(accels) > 0 and accels.max() > max_accel:
        return False
    return True