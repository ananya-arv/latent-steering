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
    find steering vec w in latent space that pts toward high risk
    """
    scaler   = StandardScaler()
    z_scaled = scaler.fit_transform(latents)

    lo_thresh = np.percentile(risks, percentile)
    hi_thresh = np.percentile(risks, 100 - percentile)
    lo_mask   = risks <= lo_thresh
    hi_mask   = risks >= hi_thresh

    w_mean = z_scaled[hi_mask].mean(0) - z_scaled[lo_mask].mean(0)
    w_mean = w_mean / (np.linalg.norm(w_mean) + 1e-8)
    probe  = Ridge(alpha=1.0)
    probe.fit(z_scaled, risks)
    r2     = probe.score(z_scaled, risks)

    w_probe = probe.coef_.copy()
    w_probe = w_probe / (np.linalg.norm(w_probe) + 1e-8)

    cos_sim = float(np.dot(w_mean, w_probe))
    print(f"  Linear R²:                    {r2:.4f}  "
          f"({'good linear structure' if r2 > 0.25 else 'weak linear structure'})")
    print(f"  cos_sim(w_mean, w_probe):     {cos_sim:.4f}  "
          f"({'vectors agree' if cos_sim > 0.5 else 'vectors disagree'})")
    print(f"  High-risk samples used:       {hi_mask.sum()}")
    print(f"  Low-risk  samples used:       {lo_mask.sum()}")

    return w_mean, w_probe, r2, scaler


def steer_and_decode(model, obs: torch.Tensor, w: np.ndarray,
                     alpha: float, scaler: StandardScaler,
                     device: str = "cuda"):
    """
    generate a trajectory steered alpha steps along w

    alpha > 0 steers toward high-risk behavior.
    alpha < 0 steers toward low-risk behavior.
    alpha = 0 reproduces the original prediction.
    """
    model.eval()
    with torch.no_grad():
        z = model.encode(obs.to(device)).cpu().numpy()   #(1, latent_dim)

        z_scaled  = scaler.transform(z)                  
        z_steered = z_scaled + alpha * w                 
        z_unscaled = scaler.inverse_transform(z_steered) 

        z_t  = torch.tensor(z_unscaled, dtype=torch.float32).to(device)
        pred = model.decode(z_t).cpu().numpy()[0]        #(pred_len, 2)

    return pred, z[0], z_unscaled[0]


def is_plausible(traj_xy: np.ndarray,
                 dt: float = 1.0 / 30,
                 max_speed: float = 6.0,
                 max_accel: float = 3.0) -> bool:
    """
    need to verify if satisfies pedestrian physical constraints

    chosen constraints:
        --> max_speed: 6.0 m/s  {sprinting, rare but physically possible}
        --> max_accel: 3.0 m/s2 {hard stop from walking speed}

    returns:
        True if trajectory is physically plausible, False otherwise
    """
    if len(traj_xy) < 2:
        return True

    velocities = np.diff(traj_xy, axis=0) / dt       #(T-1, 2)
    speeds     = np.sqrt((velocities ** 2).sum(-1))   #(T-1,)
    accels     = np.abs(np.diff(speeds)) / dt         #(T-2,)

    if speeds.max() > max_speed:
        return False
    if len(accels) > 0 and accels.max() > max_accel:
        return False
    return True