"""
M0 imitation trainer: train a BattleshipCNN (src/model/nets.py) to imitate the
Monte-Carlo solver's per-cell hit marginals (src/solver/solver.py).

Only the *policy head* is supervised (labels are per-cell marginals in [0, 1]).
The value head is initialized but left untrained here; PPO will train it.

Usage:
    python src/model/train_imitation.py --size 10K --samples 10000 --epochs 10
    python src/model/train_imitation.py --size 1M --samples 100000 --epochs 5 --seed 7
    python src/model/train_imitation.py --smoke      # tiny end-to-end sanity run

Outputs:
    checkpoints/imitation_<size>_seed<seed>_best.pt
    checkpoints/imitation_<size>_seed<seed>_final_ep<epochs>.pt
    (checkpoint = state_dict + config: target size, actual param count, seed, lr, ...)

Per-epoch log: train loss, val loss, and top-1 agreement on *unknown* cells
(diagnostic only -- per AGENTS.md, fidelity is not a headline metric).
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Make sibling modules importable when run as a plain script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, ".."))
for _sub in ("model", "gen", "solver"):
    _p = os.path.join(_SRC, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import build_dataset  # noqa: E402
from nets import count_params, make_net  # noqa: E402

SIZE_ALIASES = {
    "100": 100,
    "1K": 1_000,
    "10K": 10_000,
    "100K": 100_000,
    "1M": 1_000_000,
    "10M": 10_000_000,
    "100M": 100_000_000,
}


def _resolve_size(spec: str) -> int:
    s = spec.strip().upper()
    if s in SIZE_ALIASES:
        return SIZE_ALIASES[s]
    try:
        return int(spec)
    except ValueError:
        choices = [k for k, v in sorted(SIZE_ALIASES.items(), key=lambda kv: kv[1])]
        raise ValueError(f"unknown size {spec!r}; use one of {choices} or a raw int")


def _top1_agreement(policy: torch.Tensor, labels: torch.Tensor,
                    unknown: torch.Tensor) -> float:
    """Per-sample argmax agreement over *unknown* cells only (diagnostic)."""
    correct = total = 0
    for i in range(policy.shape[0]):
        mask = unknown[i] > 0.5
        if not bool(mask.any()):
            continue
        p = policy[i].masked_fill(~mask, float("-inf"))
        y = labels[i].masked_fill(~mask, float("-inf"))
        correct += int(p.argmax() == y.argmax())
        total += 1
    return correct / total if total else float("nan")


def _ckpt_payload(net, param_target, seed, lr, batch_size, num_samples,
                  solver_iters, data_seed, epoch, val_loss):
    return {
        "state_dict": net.state_dict(),
        "architecture": "BattleshipCNN",
        "param_target": param_target,
        "param_count": count_params(net),
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "solver_iters": solver_iters,
        "data_seed": data_seed,
        "epoch": epoch,
        "val_loss": val_loss,
    }


def train(param_target: int, num_samples: int, epochs: int, batch_size: int,
          lr: float, seed: int, device: str, solver_iters: int,
          cache_dir, data_seed: int, val_fraction: float = 0.1):
    random.seed(seed)
    torch.manual_seed(seed)

    # --- data ---
    print(f"[train] dataset: {num_samples} samples, solver iters={solver_iters}")
    X, Y = build_dataset(num_samples, seed=data_seed, num_solver_iters=solver_iters,
                         cache_dir=cache_dir)

    if num_samples < 2:
        raise ValueError("need at least 2 samples for a train/val split")
    n_val = min(max(1, int(num_samples * val_fraction)), num_samples - 1)
    X_train, Y_train = X[:-n_val], Y[:-n_val]
    X_val, Y_val = X[-n_val:], Y[-n_val:]
    print(f"[train] train={X_train.shape[0]}  val={X_val.shape[0]}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    vp = torch.from_numpy(X_val).to(device)
    vy = torch.from_numpy(Y_val).to(device)

    # --- model / optimizer ---
    net = make_net(param_target).to(device)
    n_params = count_params(net)
    print(f"[train] size target={param_target:,}  actual params={n_params:,}  device={device}")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # --- training loop ---
    ckpt_dir = os.path.normpath(os.path.join(_SRC, "..", "checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)
    size_name = str(param_target)
    base = os.path.join(ckpt_dir, f"imitation_{size_name}_seed{seed}")

    best_val = float("inf")
    best_path = base + "_best.pt"
    for epoch in range(1, epochs + 1):
        net.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            policy, _ = net(xb)
            loss = criterion(policy, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item() * yb.shape[0]
            n += yb.shape[0]
        train_loss = tot / max(n, 1)

        net.eval()
        with torch.no_grad():
            policy, _ = net(vp)
            val_loss = criterion(policy, vy).item()
            top1 = _top1_agreement(policy, vy, vp[:, 0])  # plane 0 = unknown cells

        line = (f"[train] epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  top1_unknown={top1:.3f}  ({time.time() - t0:.1f}s)")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(_ckpt_payload(net, param_target, seed, lr, batch_size,
                                     num_samples, solver_iters, data_seed,
                                     epoch, val_loss), best_path)
            line += "  *best*"
        print(line)

    torch.save(_ckpt_payload(net, param_target, seed, lr, batch_size,
                             num_samples, solver_iters, data_seed, epochs,
                             best_val), base + f"_final_ep{epochs}.pt")
    print(f"[train] done. best val_loss={best_val:.4f}")
    print(f"[train] checkpoints: {best_path}, {base}_final_ep{epochs}.pt")
    return best_val


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="M0 imitation CNN trainer")
    p.add_argument("--size", default="10K",
                   help="param target: 100/1K/10K/100K/1M/10M/100M (or raw int)")
    p.add_argument("--samples", type=int, default=10_000,
                   help="dataset size (default 10000)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42, help="torch/training seed")
    p.add_argument("--data-seed", type=int, default=42, help="seed for dataset generation")
    p.add_argument("--solver-iters", type=int, default=2000,
                   help="Monte-Carlo solver iterations per label")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--cache-dir",
                   default=os.path.normpath(os.path.join(_SRC, "..", "data")))
    p.add_argument("--no-cache", action="store_true",
                   help="skip loading/saving the dataset cache")
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end run (10K net, 64 samples, 3 epochs, no cache)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.smoke:
        args.size, args.samples, args.epochs, args.batch_size = "10K", 64, 3, 16
        args.solver_iters, args.no_cache = 500, True
        print("[smoke] tiny end-to-end run: size=10K samples=64 epochs=3 iters=500")

    param_target = _resolve_size(args.size)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train(param_target=param_target,
          num_samples=args.samples,
          epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr,
          seed=args.seed,
          device=device,
          solver_iters=args.solver_iters,
          cache_dir=None if args.no_cache else args.cache_dir,
          data_seed=args.data_seed,
          val_fraction=args.val_fraction)


if __name__ == "__main__":
    main()

