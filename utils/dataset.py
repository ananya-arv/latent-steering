"""
Raw SDD annotation format:
    track_id  x_min  y_min  x_max  y_max  frame_id  lost  occluded  generated  label

--> compute center (x,y) from bounding box, convert pixels→meters, estimate velocities via finite differences, then slide windows.
"""

import os
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ── Physical constants (from Robicquet et al. ECCV 2016) ─────────────
PIXEL_TO_METER = 0.0375
FPS            = 30
DT             = 1.0 / FPS

MAX_SPEED      = 6.0    # m/s
MAX_ACCEL      = 4.0    # m/s²
MAX_TURN_RATE  = 5.0    # rad/s
MAX_STEP_DIST  = 0.5    # m/frame
MIN_SPEED_MOVING  = 0.3   
RISK_SPEED_CEIL   = 4.0   
RISK_ACCEL_CEIL   = 2.0 
RISK_TURN_CEIL    = 1.5 


SDD_COLS = ["track_id", "x_min", "y_min", "x_max", "y_max",
            "frame_id", "lost", "occluded", "generated", "label"]

def load_sdd_txt(filepath: str, agent_type: str = "Pedestrian") -> dict:
    """
    load one raw SDD annotations.txt file.

    Returns:
        agents: dict  track_id → np.array (T, 4) = [x, y, vx, vy] in meters
    """
    try:
        df = pd.read_csv(filepath, sep=" ", header=None, names=SDD_COLS)
    except Exception:
        return {}

    df["label"] = df["label"].str.replace('"', '').str.strip()
    df = df[(df["label"] == agent_type) & (df["lost"] == 0)]
    if df.empty:
        return {}

    df = df.copy()
    df["x"] = ((df["x_min"] + df["x_max"]) / 2.0) * PIXEL_TO_METER
    df["y"] = ((df["y_min"] + df["y_max"]) / 2.0) * PIXEL_TO_METER

    agents = {}
    for agent_id, group in df.groupby("track_id"):
        group = group.sort_values("frame_id").reset_index(drop=True)
        xy    = group[["x", "y"]].values.astype(np.float32)

        if len(xy) < 5:
            continue

        vxy = np.gradient(xy, DT, axis=0)
        agents[agent_id] = np.concatenate([xy, vxy], axis=1)  # (T, 4)

    return agents



def extract_windows(
    agents: dict,
    obs_len: int  = 15,
    pred_len: int = 25,
    stride: int   = 5,
) -> list:
    """
    sliding window over each agent trajectory.

    At 30 Hz:
        obs_len=15  → 0.5 s observation
        pred_len=25 → 0.83 s prediction

    Returns list of (obs, pred) tuples, each np.array shape (T, 4).
    """
    windows = []
    total   = obs_len + pred_len

    for traj in agents.values():
        if len(traj) < total:
            continue
        for start in range(0, len(traj) - total + 1, stride):
            obs  = traj[start : start + obs_len]
            pred = traj[start + obs_len : start + total]
            windows.append((obs, pred))

    return windows


def normalize_window(obs: np.ndarray, pred: np.ndarray):
    """
    with last observed position → origin and last observed heading  → +x axis

    model can learn relative motion patterns
    """
    origin = obs[-1, :2].copy()
    vx, vy = float(obs[-1, 2]), float(obs[-1, 3])
    speed  = np.sqrt(vx**2 + vy**2)

    if speed < 1e-4:
        cos_t, sin_t = 1.0, 0.0
    else:
        cos_t =  vx / speed
        sin_t = -vy / speed

    def transform(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        out[:, 0] -= origin[0]
        out[:, 1] -= origin[1]
        x_r = cos_t * out[:, 0] - sin_t * out[:, 1]
        y_r = sin_t * out[:, 0] + cos_t * out[:, 1]
        out[:, 0], out[:, 1] = x_r, y_r
        vx_r = cos_t * out[:, 2] - sin_t * out[:, 3]
        vy_r = sin_t * out[:, 2] + cos_t * out[:, 3]
        out[:, 2], out[:, 3] = vx_r, vy_r
        return out

    return transform(obs), transform(pred)



def compute_risk_score(traj: np.ndarray) -> float:
    """
    scalar risk score in [0, 1] for a pedestrian trajectory window.

    Three components:
        speed (running near traffic is unusual/dangerous)
        accel  (sudden speed changes = unpredictable)
        turn rate  (erratic direction changes = collision risk)

    Args:
        traj: (T, 4) = [x, y, vx, vy]
    """
    speeds = np.sqrt(traj[:, 2]**2 + traj[:, 3]**2)

    if speeds.mean() < MIN_SPEED_MOVING:
        return -1.0

    accels    = np.abs(np.diff(speeds)) / DT
    headings  = np.arctan2(traj[:, 3], traj[:, 2])
    d_heading = np.abs(np.diff(headings))
    d_heading = np.minimum(d_heading, 2 * np.pi - d_heading)
    turn_rate = d_heading / DT

    r_speed = float(np.clip((speeds.max() - MIN_SPEED_MOVING) /
                            (RISK_SPEED_CEIL - MIN_SPEED_MOVING), 0.0, 1.0))
    r_accel = float(np.clip(accels.max() / RISK_ACCEL_CEIL, 0.0, 1.0)) \
              if len(accels) > 0 else 0.0
    r_turn  = float(np.clip(turn_rate.max() / RISK_TURN_CEIL, 0.0, 1.0)) \
              if len(turn_rate) > 0 else 0.0

    return 0.35 * r_speed + 0.35 * r_accel + 0.30 * r_turn



class TrajectoryDataset(Dataset):
    def __init__(self, windows: list):
        self.data = []
        for obs, pred in windows:
            obs_n, pred_n = normalize_window(obs, pred)
            risk          = compute_risk_score(obs_n)
            self.data.append((
                torch.tensor(obs_n,  dtype=torch.float32),
                torch.tensor(pred_n, dtype=torch.float32),
                torch.tensor(risk,   dtype=torch.float32),
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def get_dataloaders(
    data_dir: str,
    obs_len: int          = 15,
    pred_len: int         = 25,
    stride: int           = 5,
    batch_size: int       = 128,
    rare_threshold: float = 0.85,
    seed: int             = 42,
):
    """
    load all the SDD pedestrian annotations and return four DataLoaders.

    Returns: train_loader, val_loader, test_loader, rare_loader
    """
    np.random.seed(seed)

    txt_files = glob.glob(os.path.join(data_dir, "**/annotations.txt"), recursive=True)
    assert len(txt_files) > 0, (
        f"No annotations.txt found in {data_dir}\n"
        f"Expected structure: {data_dir}/scene/videoN/annotations.txt"
    )
    print(f"Found {len(txt_files)} annotation files")

    all_windows = []
    for f in txt_files:
        agents  = load_sdd_txt(f, agent_type="Pedestrian")
        windows = extract_windows(agents, obs_len, pred_len, stride)
        all_windows.extend(windows)

    print(f"Total windows: {len(all_windows)}")
    assert len(all_windows) > 0, "No windows extracted — check agent_type filter"

    risks = np.array([
        compute_risk_score(normalize_window(w[0], w[1])[0])
        for w in all_windows
    ])

    valid_mask  = risks >= 0
    all_windows = [all_windows[i] for i in np.where(valid_mask)[0]]
    risks       = risks[valid_mask]
    print(f"After removing stationary: {len(all_windows)} windows")

    print(
        f"Risk — mean: {risks.mean():.3f}  std: {risks.std():.3f}  "
        f"p60: {np.percentile(risks,60):.3f}  p80: {np.percentile(risks,80):.3f}"
    )

    rare_mask = risks >= rare_threshold
    rare      = [all_windows[i] for i in np.where(rare_mask)[0]]
    normal    = [all_windows[i] for i in np.where(~rare_mask)[0]]
    print(f"Normal: {len(normal)} | Rare (held-out): {len(rare)}")

    idx    = np.random.permutation(len(normal))
    normal = [normal[i] for i in idx]
    n      = len(normal)
    n_val  = int(n * 0.15)
    n_test = int(n * 0.15)

    val_w   = normal[:n_val]
    test_w  = normal[n_val : n_val + n_test]
    train_w = normal[n_val + n_test :]

    print(f"Train: {len(train_w)} | Val: {len(val_w)} | Test: {len(test_w)} | Rare: {len(rare)}")

    def make_loader(ds, shuffle):
        return DataLoader(
            TrajectoryDataset(ds),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=2,
            pin_memory=True,
        )

    return (
        make_loader(train_w, shuffle=True),
        make_loader(val_w,   shuffle=False),
        make_loader(test_w,  shuffle=False),
        make_loader(rare,    shuffle=False),
    )


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "SDD/annotations/"
    loaders  = get_dataloaders(data_dir, batch_size=64)

    for obs, pred, risk in loaders[0]:
        print(f"obs:  {obs.shape}   expect (64, 15, 4)")
        print(f"pred: {pred.shape}  expect (64, 25, 4)")
        print(f"risk: {risk.shape}  range [{risk.min():.3f}, {risk.max():.3f}]")
        break
    print("Data pipeline OK")