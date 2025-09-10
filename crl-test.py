#!/usr/bin/env python3
"""
CRL over (start, goal) pairs with a single pair-encoder f(s, g) -> R^repr_dim.

What this does
--------------
- Builds a four-room grid or a small tree environment (same structure as before).
- Collects random trajectories.
- Samples anchors (s,g) from a trajectory and positives (a,b) uniformly within [i..j] on the
  same trajectory: i <= a <= b <= j.
- Encodes each ordered pair with a SINGLE network f(s,g) (no h/g roles).
- Trains with an InfoNCE-style loss using pairwise distances: diagonal = positive, others = negatives.
- Prints metrics and saves parameters.

Run:
  python crl_pairs_single_encoder.py

Dependencies:
  jax, jaxlib, haiku, optax, numpy, tqdm
"""

from typing import Tuple
import pickle
import numpy as np
from tqdm import trange
import functools
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from jax.lib import xla_bridge
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
env_type = "grid"          # "grid" or "tree"
tree_depth_edges = 3       # for "tree": levels = depth_edges + 1, states = 2^(levels)-1

height = 10                # used if env_type == "grid"
width = 10

num_traj = 10
max_episode_steps = 10      # trajectory length = max_episode_steps + 1 states
batch_size = 256
learning_rate = 3e-3
train_steps = 500
repr_dim = 32              # output dim of the single pair encoder
loss_direction = "backward"  # "forward" | "backward" | "both"
use_hyperbolic = True  # False = Euclidean (original), True = Poincaré ball (c=1)
# ------------------------------------------------------------
# Environment (Gridworld OR Tree)
# Exposes: num_states, empty_states, step(s,a), is_empty(s), num_actions
# ------------------------------------------------------------
print(f"Using {'hyperbolic' if use_hyperbolic else 'Euclidean'} space with repr_dim={repr_dim}")

if env_type == "grid":
    def build_walls(h: int, w: int, door_frac: float = 0.2) -> np.ndarray:
        """
        Build four-room walls for an h x w grid.
        door_frac: fraction of grid dimension used for door openings (default 20%).
        """
        walls = np.zeros((h, w), dtype=int)

        # Horizontal wall in the middle
        walls[h // 2, :] = 1
        # Vertical wall in the middle
        walls[:, w // 2] = 1

        # Door size proportional to grid size
        door_h = max(1, int(h * door_frac))
        door_w = max(1, int(w * door_frac))

        # Horizontal wall doors (openings in vertical barrier)
        door_rows = np.arange(-door_h // 2, door_h // 2) + h // 4
        walls[h // 2, np.clip(door_rows, 0, w - 1)] = 0
        door_rows = np.arange(-door_h // 2, door_h // 2) + 3 * h // 4
        walls[h // 2, np.clip(door_rows, 0, w - 1)] = 0

        # Vertical wall doors (openings in horizontal barrier)
        door_cols = np.arange(-door_w // 2, door_w // 2) + w // 4
        walls[np.clip(door_cols, 0, h - 1), w // 2] = 0
        door_cols = np.arange(-door_w // 2, door_w // 2) + 3 * w // 4
        walls[np.clip(door_cols, 0, h - 1), w // 2] = 0

        return walls


    walls = build_walls(height, width)
    num_states = height * width
    empty_states = np.where(walls.flatten() == 0)[0]
    num_actions = 5  # stay, down, up, right, left

    def step(s: int, a: int) -> int:
        di, dj = np.array([[0, 0],
                           [1, 0],
                           [-1, 0],
                           [0, 1],
                           [0, -1]])[a]
        i, j = np.unravel_index(s, walls.shape)
        ni, nj = i, j
        if 0 <= i + di < height and 0 <= j + dj < width and walls[i + di, j + dj] == 0:
            ni, nj = i + di, j + dj
        return np.ravel_multi_index((ni, nj), walls.shape)

    def is_empty(s: int) -> bool:
        i, j = np.unravel_index(s, walls.shape)
        return walls[i, j] == 0

else:
    # Full binary tree: nodes in heap/level order
    levels = tree_depth_edges + 1
    num_states = (1 << levels) - 1
    empty_states = np.arange(num_states, dtype=np.int32)
    num_actions = 3  # 0=up, 1=left, 2=right

    parents = np.full(num_states, -1, dtype=np.int32)
    children = np.full((num_states, 2), -1, dtype=np.int32)
    for i in range(num_states):
        l, r = 2 * i + 1, 2 * i + 2
        if l < num_states:
            children[i, 0] = l
            parents[l] = i
        if r < num_states:
            children[i, 1] = r
            parents[r] = i

    def step(s: int, a: int) -> int:
        if a == 0:      # up
            p = parents[s]
            return int(p) if p != -1 else s
        elif a == 1:    # left
            c = children[s, 0]
            return int(c) if c != -1 else s
        else:           # right
            c = children[s, 1]
            return int(c) if c != -1 else s

    def is_empty(s: int) -> bool:
        return True




def _build_adjacency():
    """Return (adj, valid_mask).
       adj: [num_states, max_deg] int32 with -1 for padding
       valid_mask: [num_states, max_deg] bool
    """
    if env_type == "grid":
        # 4-neighborhood (no stay)
        max_deg = 4
        moves = [1, 2, 3, 4]  # down, up, right, left
        adj_np = -np.ones((num_states, max_deg), dtype=np.int32)
        for s in range(num_states):
            i, j = np.unravel_index(s, (height, width))
            nbrs = []
            for a in moves:
                ns = step(s, a)
                if ns != s:
                    nbrs.append(ns)
            adj_np[s, :len(nbrs)] = np.array(nbrs, dtype=np.int32)
        valid_np = adj_np >= 0
    else:
        max_deg = 3
        adj_np = -np.ones((num_states, max_deg), dtype=np.int32)
        valid_np = np.zeros((num_states, max_deg), dtype=bool)
        for s in range(num_states):
            tmp = []
            p = int(parents[s])
            if p != -1: tmp.append(p)
            l = int(children[s, 0]); r = int(children[s, 1])
            if l != -1: tmp.append(l)
            if r != -1: tmp.append(r)
            if tmp:
                adj_np[s, :len(tmp)] = np.array(tmp, dtype=np.int32)
                valid_np[s, :len(tmp)] = True

    return jnp.array(adj_np), jnp.array(valid_np)

ADJ, ADJ_VALID = _build_adjacency()  # shapes: [num_states, max_deg]


def _pair_embed_apply(params, s_idx_1d: jnp.ndarray, g_idx_1d: jnp.ndarray) -> jnp.ndarray:
    """Vectorized f(s,g): s_idx_1d, g_idx_1d both [E] -> embeddings [E, repr_dim]."""
    return pair_encoder.apply(params, s_idx_1d[:, None], g_idx_1d[:, None])

def _masked_softmax(logits, mask, tau):
    # mask: True where valid
    logits = logits / tau
    logits = jnp.where(mask, logits, -jnp.inf)
    return jax.nn.softmax(logits, axis=-1)

# ------------------------------------------------------------
# Trajectory collection
# ------------------------------------------------------------
def collect_trajectories(n_traj: int, max_steps: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in trange(n_traj, desc="Collecting trajectories"):
        s0 = int(rng.choice(empty_states))
        assert is_empty(s0)
        traj = [s0]
        for __ in range(max_steps):
            a = int(rng.integers(0, num_actions))
            ns = step(traj[-1], a)
            if not is_empty(ns):
                raise RuntimeError("Hit a wall; should not happen.")
            traj.append(ns)
        trajs.append(traj)
    return np.array(trajs, dtype=np.int32)  # [num_traj, max_episode_steps+1]

# ------------------------------------------------------------
# State features
# ------------------------------------------------------------
def _state_to_xy(idx: jnp.ndarray) -> jnp.ndarray:
    """
    Map flattened state index to a 2D coordinate feature.
    idx shape: [..., 1] -> returns [..., 2] (float32)
    """
    if env_type == "grid":
        i = idx // width
        j = idx % width
        return jnp.concatenate([i, j], axis=-1).astype(jnp.float32)
    else:
        # Tree: (level, normalized_position_within_level)
        one = jnp.ones_like(idx, dtype=idx.dtype)
        level = jnp.floor(jnp.log2(idx + one)).astype(jnp.int32)
        level_start = (jnp.power(2, level) - one).astype(jnp.int32)  # 2^level - 1
        pos = (idx - level_start).astype(jnp.int32)
        level_width = jnp.maximum(jnp.power(2, level), one).astype(jnp.float32)
        x = pos.astype(jnp.float32) / level_width  # [0,1)
        y = level.astype(jnp.float32)
        return jnp.concatenate([y, x], axis=-1)

# ------------------------------------------------------------
# Single pair encoder f(s, g) -> R^repr_dim
# ------------------------------------------------------------
def _pair_encoder(s_idx: jnp.ndarray, g_idx: jnp.ndarray) -> jnp.ndarray:
    """
    Encode an ordered pair (s,g) as a single vector z in R^repr_dim.

    Inputs:
      s_idx: [..., 1] int32
      g_idx: [..., 1] int32
    Returns:
      z:    [..., repr_dim] float32
    """
    xs = _state_to_xy(s_idx)  # [..., 2]
    xg = _state_to_xy(g_idx)  # [..., 2]
    # Features: concat of both and their difference (helps model relative geometry)
    feats = jnp.concatenate([xs, xg], axis=-1)  # [..., 6]

    net = hk.nets.MLP([256, 256, repr_dim], activation=jax.nn.relu, name="pair_encoder")
    v = net(feats)
    if use_hyperbolic:
        # keep a margin from the boundary to avoid denom ≈ 0
        eps = 1e-4
        r = jnp.linalg.norm(v, axis=-1, keepdims=True)
        v = (1.0 - eps) * jnp.tanh(r) * (v / (r + 1e-12))
    return v

pair_encoder = hk.without_apply_rng(hk.transform(_pair_encoder))

# ------------------------------------------------------------
# Loss: InfoNCE over pair embeddings
# ------------------------------------------------------------
## this returns the l2 distance matrix
def _pairwise_sqdist(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    If use_hyperbolic: return Poincaré (c=1) distance with safety guards.
    Else: squared Euclidean (original).
    """
    if use_hyperbolic:
        a2 = jnp.sum(A * A, axis=1, keepdims=True)            # [B,1]
        b2 = jnp.sum(B * B, axis=1, keepdims=True).T           # [1,B]
        sq = jnp.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)       # ||x-y||^2

        # denominator (1 - ||x||^2)(1 - ||y||^2) with floor to avoid blow-up
        min_denom = 1e-2
        denom = jnp.maximum((1.0 - a2) * (1.0 - b2), min_denom)

        # argument to arcosh: must be >= 1
        z = 1.0 + 2.0 * sq / denom
        z = jnp.maximum(z, 1.0 + 1e-7)

        
            # Print only if suspicious
        jax.debug.print(
            "hyp dbg: min(1-||A||^2)={} min(1-||B||^2)={} min(z)={} max(z)={}",
            jnp.round(jnp.min(1.0 - a2), 6),
            jnp.round(jnp.min(1.0 - b2), 6),
            jnp.round(jnp.min(z), 6),
            jnp.round(jnp.max(z), 6),
        )

        return jnp.arccosh(z)  # hyperbolic distance (not squared)
    else:
        Aa = jnp.sum(A * A, axis=1, keepdims=True)           # [B,1]
        Bb = jnp.sum(B * B, axis=1, keepdims=True).T         # [1,B]
        D = Aa + Bb - 2.0 * (A @ B.T)
        return jnp.maximum(D, 0.0)

def loss_fn(params, s_anchor, g_anchor, s_pos, g_pos):
    """
    Build embeddings for anchors and positives, compute full distance matrix,
    then apply an InfoNCE-style objective:
      loss = mean( diag(D) + logsumexp(-D, axis=dir) )
    """
    Z_anchor = pair_encoder.apply(params, s_anchor, g_anchor)  # [B, repr_dim]
    Z_pos    = pair_encoder.apply(params, s_pos,    g_pos)     # [B, repr_dim]

    D = _pairwise_sqdist(Z_anchor, Z_pos)  # [B,B]; diagonal entries are positives
    l_align = jnp.diag(D)

    if loss_direction == "forward":
        l_unif = jax.nn.logsumexp(-D, axis=1)
    elif loss_direction == "backward":
        l_unif = jax.nn.logsumexp(-D.T, axis=1)
    else:
        l_unif = 0.5 * (jax.nn.logsumexp(-D, axis=1) + jax.nn.logsumexp(-D.T, axis=1))

    loss = (l_align + l_unif).mean()
    acc = jnp.mean(jnp.argmin(D, axis=1) == jnp.arange(D.shape[0]))
    avg_pos = jnp.mean(jnp.diag(D))
    avg_neg = jnp.mean(D - jnp.eye(batch_size))  # mask diagonal
    return loss, (avg_neg, avg_pos, acc)

value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

# ------------------------------------------------------------
# Sampler: anchors (s,g) and positives (a,b) with i<=a<=b<=j on same traj
# ------------------------------------------------------------
@jax.jit
def get_batch_pairs(rng_key, ds_obs: jnp.ndarray):
    """
    ds_obs: [N_traj, L], L = max_episode_steps + 1
    Returns s_anchor, g_anchor, s_pos, g_pos with shape [B,1] (int32).
    """
    B = batch_size
    N = ds_obs.shape[0]
    L = ds_obs.shape[1]

    k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)

    # Choose trajectories from first 80% for training
    max_train = (N * 4) // 5
    traj_idx = jax.random.randint(k1, (B,), 0, max_train)  # [B]

    # Sample two time indices and order them (ensure i < j)
    t0 = jax.random.randint(k2, (B,), 0, L)
    t1 = jax.random.randint(k3, (B,), 0, L)
    i = jnp.minimum(t0, t1)
    j = jnp.maximum(t0, t1)
    equal = (i == j)
    i = jnp.where(equal, jnp.maximum(i - 1, 0), i)
    j = jnp.where(equal, jnp.minimum(j + 1, L - 1), j)
    i = jnp.minimum(i, j - 1)

    # Positive pair (a,b), uniform with i <= a <= b <= j
    k_a = jax.random.uniform(k4, (B,))
    span_ij = (j - i)  # >= 1
    a = i + jnp.floor(k_a * (span_ij + 1.000001)).astype(jnp.int32)
    a = jnp.minimum(a, j)

    k_b = jax.random.uniform(k5, (B,))
    span_ab = (j - a)
    b = a + jnp.floor(k_b * (span_ab + 1.000001)).astype(jnp.int32)
    b = jnp.minimum(b, j)

    # Gather states
    s_anchor = ds_obs[traj_idx, i][:, None]  # [B,1]
    g_anchor = ds_obs[traj_idx, j][:, None]
    s_pos    = ds_obs[traj_idx, a][:, None]
    g_pos    = ds_obs[traj_idx, b][:, None]
    # jax.debug.print("i={} j={} a={} b={}", i[0], j[0], a[0], b[0])

    return s_anchor, g_anchor, s_pos, g_pos

@jax.jit
def train_step(params, opt_state, rng_key, ds_obs):
    s_anchor, g_anchor, s_pos, g_pos = get_batch_pairs(rng_key, ds_obs)
    (loss, (l_unif, l_align, acc)), grads = value_and_grad(params, s_anchor, g_anchor, s_pos, g_pos)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, (l_unif, l_align, acc)


@functools.partial(jax.jit, static_argnames=('max_hops', 'tau'))
def jax_test_success(params, s0: jnp.ndarray, g: jnp.ndarray, *, max_hops: int = 200, tau: float = 1.0, key: jnp.ndarray = None):
    """
    Batched soft rollout success.
    s0, g: [E] int32 (E episodes in parallel)
    Returns success_rate (scalar float32).
    """
    E = s0.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)

    def step_fn(carry, _):
        key, s, done = carry  # s: [E], done: [E] bool

        # f(s,g)
        z_sg = _pair_embed_apply(params, s, g)  # [E, D]

        # neighbors for each s
        nbrs = ADJ[s]              # [E, max_deg]
        valid = ADJ_VALID[s]       # [E, max_deg]
        # For invalid slots, put a dummy index 0; mask will zero them out.
        nbrs_safe = jnp.where(valid, nbrs, 0)

        # f(w,g) for all neighbors (flatten then reshape)
        Emax = nbrs_safe.shape[0] * nbrs_safe.shape[1]
        flat_w = nbrs_safe.reshape(-1)          # [E*max_deg]
        flat_g = jnp.repeat(g, nbrs_safe.shape[1])  # [E*max_deg]
        z_wg = _pair_embed_apply(params, flat_w, flat_g).reshape(nbrs_safe.shape[0], nbrs_safe.shape[1], -1)  # [E,max_deg,D]

        # distances to anchor z_sg
        # broadcast z_sg: [E,1,D]
        d = jnp.sum((z_wg - z_sg[:, None, :])**2, axis=-1)  # [E, max_deg]
        logits = -d
        probs = _masked_softmax(logits, valid, tau)         # [E, max_deg]

        # sample next neighbor index for each episode
        key, sub = jax.random.split(key)
        # categorical expects logits; use log(probs) (safe because masked zeros -> -inf)
        idx = jax.random.categorical(sub, jnp.log(probs + 1e-20), axis=-1)  # [E]
        w = jnp.take_along_axis(nbrs_safe, idx[:, None], axis=1).squeeze(1) # [E]

        # update s unless already done
        s_next = jnp.where(done, s, w)
        done_next = jnp.logical_or(done, s_next == g)
        return (key, s_next, done_next), None

    # scan over hops
    init = (key, s0, jnp.zeros_like(s0, dtype=bool))
    (key, sT, doneT), _ = lax.scan(step_fn, init, xs=None, length=max_hops)

    success = doneT.astype(jnp.float32)  # [E]
    return jnp.mean(success)             # scalar

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print(f"Env: {env_type} | num_states={num_states} | num_actions={num_actions}")
    print(f"JAX backend platform: {xla_bridge.get_backend().platform}")

    # Data
    print("Collecting trajectories...")
    ds_obs_np = collect_trajectories(num_traj, max_episode_steps, seed=0)
    ds_obs = jnp.array(ds_obs_np)  # [N, L]

    # Init encoder params
    key = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, 1), dtype=jnp.int32)
    params = pair_encoder.init(key, dummy, dummy)

    # Optimizer
    global optimizer
    
    if use_hyperbolic:
        learning_rate = 3e-4  # 10× smaller
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate),
        )
    else:
        optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Train
    print("Training...")
    rng = jax.random.PRNGKey(42)
    for t in range(train_steps):
        rng, sub = jax.random.split(rng)
        params, opt_state, loss, (avg_neg, avg_pos, acc) = train_step(params, opt_state, sub, ds_obs)
        if (t + 1) % 50000 == 0:
            # run 5 test episodes in parallel on-device
            E = 10
            key_eval = jax.random.PRNGKey(t)
            s0 = jax.random.randint(key_eval, (E,), 0, num_states, dtype=jnp.int32)
            # ensure start/goal are empty states (for grid walls); sample from empty_states table
            # (fast path) – pick from empty_states via indexing
            idx_key = jax.random.PRNGKey(t + 1)
            eidx_s = jax.random.randint(idx_key, (E,), 0, empty_states.shape[0])
            eidx_g = jax.random.randint(idx_key, (E,), 0, empty_states.shape[0])
            s0 = jnp.array(empty_states)[eidx_s]
            g  = jnp.array(empty_states)[eidx_g]
            # avoid s==g (optional)
            g = jnp.where(s0 == g, (g + 1) % num_states, g)

            test_rate = jax_test_success(params, s0, g, max_hops=200, tau=1.0, key=key_eval)

            print(
                f"step={t+1:6d}  loss={float(loss):.4f}  "
                f"avg_neg={float(avg_neg):.4f}  avg_pos={float(avg_pos):.4f}  acc={float(acc):.4f}  "
                f"test_success@5={float(test_rate):.3f}"
            )
            params_path = f"params/params_pairencoder_{env_type}_{tree_depth_edges}_{use_hyperbolic}_{repr_dim}_{t+1}.pkl"
            with open(params_path, "wb") as f:
                pickle.dump(params, f)
            print(f"Saved Haiku params to {params_path}")

    print("Done.")

    # Save params
    

if __name__ == "__main__":
    main()
