""" 
Partial-game state generator for Battleship training data.

Given a seeded legal full board (recycled from ``generate_board``), simulates a
partial game of shots against it and returns the resulting **known-info state**:

    state cell codes:
        0 = unknown        (never fired at)
        1 = miss           (water revealed)
        2 = hit            (active, not-yet-sunk ship cell)
        3 = sunk           (a fully-sunk ship, all cells revealed)

Alongside the state we return the *remaining fleet* (lengths of ships not yet
sunk) and the *truth* board, so downstream consumers (exact probabilistic
solver -> teacher labels, and the imitation CNN) have everything needed.

Shot selection is deterministic for a given seed: ``generate_board`` seeds
``numpy``'s global RNG with ``input_seed``, and shot simulation continues on
that same stream, so the whole pipeline is reproducible per seed.
"""

import numpy as np
import cupy as cp

from collections import deque
from generate_board import generate_board, SHIP_LENGTHS, BOARD_SIZE

# State cell codes
UNKNOWN = 0
MISS = 1
HIT = 2
SUNK = 3

_POLICIES = ("expand", "random") # Policies for turns after revealing a ship, expand on the ship or shoot randomly

_ORTHO = ((-1, 0), (1, 0), (0, -1), (0, 1))


def generate_partial_state(input_seed, num_shots=None, policy="expand"):
    """Return a partial-game state derived from a seeded legal board.

    Parameters
    ----------
    input_seed : int
        Seed for board generation and shot simulation.
    num_shots : int or None
        Number of shots to attempt against the board. ``None`` draws a uniform
        integer from 0..100 so training sees a spread of game progressions
        (early, mid, near-complete). The simulation stops early if the board
        runs out of unknown cells (i.e. the game ends).
    policy : "expand" or "random"
        Shot-selection heuristic. ``"expand"`` fires at random unknown cells and,
        after a hit, probes the four orthogonal neighbours to try to sink ships
        (realistic games with hits/sunk). ``"random"`` fires at uniformly random
        unknown cells only.

    Returns
    -------
    dict with:
        state     : cupy int array, 0=unknown 1=miss 2=hit 3=sunk
        truth     : cupy full board (for teacher labels / verification)
        remaining : tuple of lengths of ships not yet sunk
        sunk      : numpy bool array, which ships (by 1-based id) are sunk
        num_shots : the requested shot count (actual fired may be less)
    """
    if policy not in _POLICIES:
        raise ValueError(f"policy must be one of {_POLICIES}, got policy: {policy!r}")

    board = generate_board(input_seed)  # seeds numpy deterministically
    rng = np.random

    if num_shots is None:
        num_shots = int(rng.randint(0, BOARD_SIZE * BOARD_SIZE + 1))

    state, sunk = _simulate_shots(cp.asnumpy(board), num_shots, policy, rng)
    remaining = tuple(length for length, sunk_here in zip(SHIP_LENGTHS, sunk)
                      if not sunk_here)

    return {
        "state": cp.asarray(state),
        "truth": board,
        "remaining": remaining,
        "sunk": sunk,
        "num_shots": num_shots,
    }


def _simulate_shots(board, num_shots, policy, rng):
    """Fire up to ``num_shots`` against ``board``, returning (state, sunk)."""
    n = board.shape[0]
    state = np.zeros((n, n), dtype=np.int32)
    sunk = np.zeros(len(SHIP_LENGTHS), dtype=bool)

    unknown_set = {(r, c) for r in range(n) for c in range(n)}
    random_pool = list(unknown_set)
    rng.shuffle(random_pool)
    random_iter = iter(random_pool)

    probe = deque()
    shots = 0
    while shots < num_shots:
        if policy == "expand" and probe:
            r, c = probe.popleft()
        else:
            try:
                r, c = next(random_iter)
            except StopIteration:
                break  # board exhausted (game over)
        if (r, c) not in unknown_set:
            continue  # stale probe / already-fired cell

        unknown_set.discard((r, c))
        shots += 1

        ship_id = board[r, c]
        if ship_id == 0:
            state[r, c] = MISS
            continue

        state[r, c] = HIT
        if _ship_sunk(board, ship_id, state):
            cells = {tuple(p) for p in np.argwhere(board == ship_id)}
            for rr, cc in cells:
                state[rr, cc] = SUNK
            unknown_set.difference_update(cells)
            sunk[ship_id - 1] = True
            # drop any queued probes belonging to this now-sunk ship
            probe = deque((rr, cc) for (rr, cc) in probe if (rr, cc) not in cells)
        elif policy == "expand":
            for dr, dc in _ORTHO:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and (rr, cc) in unknown_set:
                    probe.append((rr, cc))

    return state, sunk


def _ship_sunk(board, ship_id, state):
    """True if every cell of ``ship_id`` is currently a hit (not yet sunk)."""
    rows, cols = np.nonzero(board == ship_id)
    return all(state[r, c] == HIT for r, c in zip(rows, cols))


def _verify_partial(state, truth, sunk):
    """Sanity-check a partial state against its truth board (mirrors _verify)."""
    state = cp.asnumpy(state) if hasattr(state, "get") else state
    truth = cp.asnumpy(truth) if hasattr(truth, "get") else truth

    fired = state != UNKNOWN
    # a miss must be water; a hit/sunk must be a ship cell
    assert np.all(truth[fired & (state == MISS)] == 0)
    assert np.all(truth[fired & (state != MISS)] > 0)

    # sunk ships: every cell revealed as SUNK; surviving ships: none as SUNK
    for ship_id in range(1, len(SHIP_LENGTHS) + 1):
        cells = np.argwhere(truth == ship_id)
        if sunk[ship_id - 1]:
            assert all(state[r, c] == SUNK for r, c in cells)
        else:
            assert all(state[r, c] != SUNK for r, c in cells)
    return True


if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1234
    n_shots = int(sys.argv[2]) if len(sys.argv) > 2 else None
    pol = sys.argv[3] if len(sys.argv) > 3 else "expand"

    game = generate_partial_state(seed, num_shots=n_shots, policy=pol)
    _verify_partial(game["state"], game["truth"], game["sunk"])

    print(f"seed={seed}  policy={pol}  shots={game['num_shots']}")
    print(f"remaining ships: {game['remaining']}")
    print("state (0=unk 1=miss 2=hit 3=sunk):")
    print(cp.asnumpy(game["state"]))
    print("truth:")
    print(cp.asnumpy(game["truth"]))
