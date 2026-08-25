""" 
Deterministic (seeded) Battleship board generator, returns GPU (cupy) arrays.

Generates a full board for standard rules (ships of lengths: 5,4,3,3,2, and no touching directly/diagonally)

Random placement is computed deterministically via numpy random seed, and then transferred to the GPU as
a ''cupy'' array via ''cp.asarray'' for downstream GPU workloads. ''cupy'' isn't required for this, so it
can be replaced fully with numpy, though it is required per pyproject.toml

The entire script could also be GPU-accelerated, but unless this step is a bottleneck, numpy is fine.
"""

import numpy as np
import cupy as cp

BOARD_SIZE = 10 # standard Battleship board
SHIP_LENGTHS = (5, 4, 3, 3, 2)  # standard Battleship fleet


def generate_board(input_seed):
    """Return a seeded, legal full board as a 2-D cupy int array.

    Cells hold ``0`` (empty water) or the 1-based ship id (1 to 5) of the ship
    occupying them. Placement is deterministic for a given seed and always
    satisfies the no-touch rule (no orthogonal or diagonal contact between
    ships).
    """
    np.random.seed(input_seed)

    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    for ship_id, length in enumerate(SHIP_LENGTHS, start=1):
        _place_ship(board, length, ship_id)
    return cp.asarray(board)


def _place_ship(board, length, ship_id):
    """Place one ship of ``length`` cells, marking them with ``ship_id``."""
    n = board.shape[0]
    for _attempt in range(10000): # 10,000 attempts to place a ship, more than enough.
        horizontal = np.random.rand() < 0.5
        if horizontal:  # horizontal
            row = np.random.randint(n)
            col = np.random.randint(n - length + 1)
        else:  # vertical
            row = np.random.randint(n - length + 1)
            col = np.random.randint(n)

        if _can_place(board, row, col, length, horizontal):
            _mark(board, row, col, length, horizontal, ship_id)
            return

    raise RuntimeError(f"failed to place ship of length {length} for seed")


def _can_place(board, row, col, length, horizontal):
    """True if a ship anchored at (row, col) stays in bounds and touches no
    existing ship (including diagonally)."""
    n = board.shape[0]
    if horizontal:
        if col < 0 or col + length > n:
            return False
        cells = ((row, col + i) for i in range(length))
    else:
        if row < 0 or row + length > n:
            return False
        cells = ((row + i, col) for i in range(length))
    return all(_cell_free(board, r, c) for r, c in cells)


def _cell_free(board, r, c):
    """True if cell (r, c) and its full 3x3 neighborhood contain no ship."""
    n = board.shape[0]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and board[rr, cc]:
                return False
    return True


def _mark(board, row, col, length, horizontal, ship_id):
    """Write the ship cells onto the board."""
    if horizontal:
        for c in range(col, col + length):
            board[row, c] = ship_id
    else:
        for r in range(row, row + length):
            board[r, col] = ship_id


def _verify(board):
    """Sanity-check a board (cupy or numpy): correct counts, no touching."""
    if hasattr(board, "get"):  # cupy -> numpy copy for inspection
        board = board.get()
    n = board.shape[0]
    counts = {ship_id: int(np.count_nonzero(board == ship_id))
              for ship_id in range(1, len(SHIP_LENGTHS) + 1)}
    assert list(counts.values()) == list(SHIP_LENGTHS), counts

    occupied = board != 0
    # no two ships touch orthogonally or diagonally
    for r in range(n - 1):
        for c in range(n - 1):
            window = occupied[r:r + 2, c:c + 2]
            if np.count_nonzero(window) > 1:
                ids = np.unique(window[window != 0])
                if len(ids) > 1:
                    raise AssertionError("ships touch at (%d,%d)" % (r, c))
    return True


if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1234
    board = generate_board(seed)  # cupy array on GPU

    _verify(board)
    print(f"seed={seed}  fleet={SHIP_LENGTHS}")
    print(cp.asnumpy(board))  # inspect on CPU

    # determinism check: same seed -> identical board
    board2 = generate_board(seed)
    assert cp.array_equal(board, board2)
    print("\ndeterminism OK (same seed reproduces identical board)")

    # spot-check: different seeds usually differ
    board3 = generate_board(seed + 1)
    print("different seed differs:", not cp.array_equal(board, board3))
