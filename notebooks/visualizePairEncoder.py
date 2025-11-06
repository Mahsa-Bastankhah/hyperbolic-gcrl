#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import torch

from torch.serialization import add_safe_globals
from hypll.tensors.manifold_tensor import ManifoldTensor
add_safe_globals([ManifoldTensor])  # allow-list for torch.load(weights_only=True)

from utils import load_model

def blank_maze(h: int, w: int):
    return np.zeros((h, w), dtype=int)

@torch.no_grad()
def pair_embed(pair_encoder, s_xy, g_xy, device):
    """
    s_xy, g_xy: np.array([row, col], dtype=float)
    Returns: pair embedding z (Tensor or ManifoldTensor) with shape [1, D]
    """
    s = torch.tensor(s_xy, device=device, dtype=torch.float32).unsqueeze(0)  # [1,2]
    g = torch.tensor(g_xy, device=device, dtype=torch.float32).unsqueeze(0)  # [1,2]
    sg = torch.cat([s, g], dim=-1)                                           # [1,4]
    return pair_encoder(sg)                                                  # [1,D]

@torch.no_grad()
def hyperbolic_norm(z, manifold):
    """
    Hyperbolic 'norm' = distance from origin in the Poincaré ball.
    If z is Euclidean (Tensor), fallback to Euclidean norm.
    """
    if isinstance(z, ManifoldTensor):
        zero = ManifoldTensor(torch.zeros_like(z.tensor), manifold=manifold)
        d0 = manifold.dist(x=z, y=zero)  # [1]
        return float(d0.item())
    else:
        return float(torch.norm(z, dim=-1).item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved_dir", type=str, default="saved_models/experiment2_hyperbolic_True_curvature_1.0_learnable_True_epochs_256_trajectories_10000_order__maze_blank_embeddingdim_2_gamma_0.1_batch_128", help="Path to saved run (contains config.json)")
    ap.add_argument("--epoch", type=int, default=60)
    ap.add_argument("--num_pairs", type=int, default=20)   # ~20 base pairs (s forward, g backward)
    ap.add_argument("--repeats", type=int, default=10)     # noise samples per pair
    ap.add_argument("--maze_h", type=int, default=1)
    ap.add_argument("--maze_w", type=int, default=11)      # 11 so g=(row,10) is valid
    ap.add_argument("--noise_std", type=float, default=0.15)  # row noise std
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)

    # --- load model + pair encoder + manifold ---
    cfg = json.load(open(os.path.join(args.saved_dir, "config.json")))
    md  = load_model(cfg, device, pretrained_path=args.saved_dir, epoch=args.epoch)
    pair_encoder = md["pair_encoder"].to(device).eval()
    manifold     = md["manifold"]

    # --- maze (only used for bounds) ---
    H, W = args.maze_h, args.maze_w
    maze = blank_maze(H, W)

    # --- construct base pairs: s moves from 0→(W-1)/2, g moves from (W-1)→(W-1)/2 ---
    # ensures g_col >= s_col and spans separations from ~W-1 down to ~0
    max_shift = (W - 1) / 2.0
    s_cols = np.linspace(0.0, max_shift, args.num_pairs, dtype=np.float32)
    g_cols = (W - 1) - s_cols

    # --- evaluate mean hyperbolic norm across noise for each base pair ---
    rows = []
    for k in range(args.num_pairs):
        s_c = float(s_cols[k])
        g_c = float(g_cols[k])
        norms = []
        d_cols = g_c - s_c  # 1D separation along the row

        for _ in range(args.repeats):
            # row noise (around the only row 0). Clip to stay within [0, H-1].
            row_noise = float(np.clip(rng.normal(loc=0.0, scale=args.noise_std), 0.0, max(0.0, H-1 - 1e-6)))

            s_xy = np.array([row_noise, s_c], dtype=np.float32)
            g_xy = np.array([row_noise, g_c], dtype=np.float32)

            z = pair_embed(pair_encoder, s_xy, g_xy, device)  # [1,D], ManifoldTensor or Tensor
            norms.append(hyperbolic_norm(z, manifold))

        mean_norm = float(np.mean(norms))
        std_norm  = float(np.std(norms))
        rows.append((k, s_c, g_c, d_cols, mean_norm, std_norm))

    # --- report ---
    print("\nIdx | s_col -> g_col | sep | mean_hyp_norm ± std")
    print("-" * 60)
    for (k, s_c, g_c, d_cols, mean_norm, std_norm) in rows:
        print(f"{k:3d} | {s_c:5.2f} -> {g_c:5.2f} | {d_cols:5.2f} | {mean_norm:.4f} ± {std_norm:.4f}")

    # optional: save CSV for later plotting
    os.makedirs("figures", exist_ok=True)
    out_csv = os.path.join("figures", f"pair_norms_W{W}_N{args.num_pairs}_rep{args.repeats}_ep{args.epoch}.csv")
    with open(out_csv, "w") as f:
        f.write("idx,s_col,g_col,separation,mean_hyp_norm,std_hyp_norm\n")
        for (k, s_c, g_c, d_cols, mean_norm, std_norm) in rows:
            f.write(f"{k},{s_c:.6f},{g_c:.6f},{d_cols:.6f},{mean_norm:.6f},{std_norm:.6f}\n")
    print(f"\nSaved CSV: {out_csv}")

if __name__ == "__main__":
    main()
