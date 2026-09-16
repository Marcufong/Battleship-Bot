"""
Imitation dataset builder for the Battleship CNN (M0).

Each sample is a (state, label) pair:
    state : float32 (9, 10, 10)  -- the CNN's input planes (see nets.py):
              planes 0-3 one-hot the state cell code, planes 4-8 flag which
              fleet ships (1-based id) are not yet sunk.
    label : float32 (10, 10)     -- the solver's per-cell marginal
              P(cell holds a remaining-ship cell | state) from src/solver.

Samples are generated deterministically: sample i uses seed = seed + i for the
partial-game state (src/gen) and the same seed for the Monte-Carlo solver, so
a given (num_samples, seed, solver_iters) dataset is fully reproducible.
Datasets are cached to .npz (under data/ by default) so repeated training runs
never re-pay the (CPU-bound) solver cost.
"""

import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)) # Make src/gen and src/solver importable when this module is used from anywhere.
for _sub in ("gen", "solver"):
    _p = os.path.normpath(os.path.join(_HERE, "..", _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_partial_board import generate_partial_state  # noqa: E402
from solver import solve  # noqa: E402
from nets import BOARD_SIZE, INPUT_CHANNELS, NUM_SHIPS, NUM_STATE_CODES  # noqa: E402


def encode_input(state, sunk) -> np.ndarray:
    """Encode a partial-game state into the CNN's 9 input planes.

    state : array (10, 10) int, cell codes 0=unknown 1=miss 2=hit 3=sunk
    sunk  : bool array (5,), True where the 1-based ship id is already sunk
    """
    state = state.get() if hasattr(state, "get") else np.asarray(state)
    x = np.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for code in range(NUM_STATE_CODES):
        x[code] = (state == code).astype(np.float32)
    for i in range(NUM_SHIPS):
        x[NUM_STATE_CODES + i] = 1.0 - float(sunk[i])  # 1.0 if ship (i+1) still afloat
    return x


def _one_sample(seed: int, num_solver_iters: int):
    """Generate one (input, label) pair; retry a new seed if the solver rejects it."""
    offset = 0
    for _attempt in range(100):
        s = seed + offset
        game = generate_partial_state(s, policy="expand")
        try:
            label = solve(game["state"], game["remaining"],
                          num_iterations=num_solver_iters, seed=s)
        except ValueError:
            # Solver saw a state it cannot satisfy (very rare); draw a new one.
            offset += 1_000_000
            continue
        y = label.get() if hasattr(label, "get") else np.asarray(label)
        y = np.asarray(y, dtype=np.float32)
        if y.shape != (BOARD_SIZE, BOARD_SIZE):
            raise RuntimeError(f"unexpected label shape {y.shape} for seed {s}")
        return encode_input(game["state"], game["sunk"]), y
    raise RuntimeError(f"could not build a sample starting from seed {seed}")


def _cache_path(cache_dir: str, seed: int, num_samples: int, num_solver_iters: int) -> str:
    return os.path.join(cache_dir,
                        f"imitation_data_seed{seed}_n{num_samples}_iters{num_solver_iters}.npz")


def build_dataset(num_samples: int, seed: int = 42, num_solver_iters: int = 2000,
                  cache_dir: str = None, use_cache: bool = True):
    """Return (X, Y): float32 arrays of shape (N, 9, 10, 10) and (N, 10, 10)."""
    path = None
    if cache_dir and use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        path = _cache_path(cache_dir, seed, num_samples, num_solver_iters)
        if os.path.exists(path):
            print(f"[dataset] loading cached dataset {path}")
            data = np.load(path)
            return data["x"], data["y"]

    x = np.empty((num_samples, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    y = np.empty((num_samples, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    t0 = time.time()
    for i in range(num_samples):
        x[i], y[i] = _one_sample(seed + i, num_solver_iters)
        if (i + 1) % 250 == 0 or i + 1 == num_samples:
            print(f"[dataset] {i + 1}/{num_samples} samples built ({time.time() - t0:.1f}s)")
    if path:
        np.savez(path, x=x, y=y)
        print(f"[dataset] cached dataset -> {path}")
    return x, y


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1234
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    X, Y = build_dataset(4, seed=seed, num_solver_iters=iters, cache_dir=None)
    assert X.shape == (4, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), X.shape
    assert Y.shape == (4, BOARD_SIZE, BOARD_SIZE), Y.shape
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    assert np.all((Y >= 0) & (Y <= 1)), "labels must be probabilities"

    # label sanity on the first sample: hit/sunk cells ~1.0/known, miss cells ~0.0
    game = generate_partial_state(seed, policy="expand")
    state = game["state"].get() if hasattr(game["state"], "get") else game["state"]
    y0 = Y[0]
    assert np.all(y0[state == 2] > 0.99), "hit cells should be ~1.0"
    assert np.all(y0[state == 1] < 0.01), "miss cells should be ~0.0"
    assert np.all(y0[state == 3] < 0.01), "sunk cells should be ~0.0"

    print("X shape", X.shape, "Y shape", Y.shape, "label mean", round(float(Y.mean()), 4))
    print("dataset.py self-check OK")
