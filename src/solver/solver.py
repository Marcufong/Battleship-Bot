"""
Probabilistic targeting solver, what the CNN is trained to imitate (M0).

Turns a partial-game state (what ``src/gen/generate_partial_board.py``
returns) into the per-cell label the CNN is trained on:

    prob[r, c] = P(cell (r, c) holds a cell of a *remaining* ship | state)

The naive implementation is, given the board state and ship state:
- randomly attempt to place any partially-hit ships into legal positions that include the partially-hit cells
- randomly place the other remaining ships, in a random order, into legal positions
- return the per-cell label for each cell as the number of times a ship was in that tile, divided by the total number of iterations

State cell codes (mirror ``src/gen/generate_partial_board.py``):
    0 = unknown, 1 = miss, 2 = hit (active, not-yet-sunk ship), 3 = sunk.

How this solver works:
---------------------------------
1. Split the HIT cells into 8-connected components. Because two *different*
   ships can never be orthogonally or diagonally adjacent, one 8-connected blob
   of hits always belonds to a single ship. (A straight ship's hits are always
   collinear, so a non-collinear blob = inconsistent state -> ValueError.)
2. Sample one random *grouping* of those components into partially-hit ships.
   Any components may be grouped together only if *ALL* their cells lie on one
   straight line, the cells between them are all unknown, the combined span fits
   the longest remaining ship, and the total number of groups does not exceed the
   total number of remaining ships. This covers both possibilities of "one long ship
   hit at several non-adjacent spots" (random shots) and the "several short ships",
   whichever the state actually encodes.
3. Assign each grouped ship a random *remaining* length (>= its span), then
   place it at a random legal position covering all of its hit cells.
4. Place the other (un-hit) remaining ships in a random order, each at a
   random legal position.
5. A position is legal if none of its cells is a miss or an occupied cell,
   and no cell of the ship touches (orthogonally or diagonally) a revealed
   ship cell — enforcing the no-touch rule.

Every iteration that successfully places all remaining ships contributes its
occupancy to a per-cell counter; the label is that counter divided by the
number of *successful* iterations (a plain Monte-Carlo estimate of the
marginal). Iterations that cannot be completed are discarded.

Usage
---
``solve(state, remaining, ...)``        -> cupy (10, 10) float32 marginal, one state.
``solve_batch(states, remainings, ...)`` -> cupy (N, 10, 10) float32 stack.

Notes
---------------------------
* Groupings are sampled uniformly over *syntactically* valid groupings, so a
  grouping that admits many consistent completions is not weighted above one
  that admits few. For a naive baseline this is fine: hit cells always land at
  ~1.0 and miss/sunk cells at 0.0 regardless, and only exotic ambiguous states
  pick up a small bias on the in-between cells.
* If a state has more than 10 scattered hit components (extremely late
  "random"-policy games), the grouping enumeration falls back to random
  sampling of groupings instead of full enumeration.
* ``solve`` raises ``ValueError`` on a state with no syntactically valid
  grouping (inconsistent with the straight-ship / no-touch rules); the dataset
  generator should catch that and skip the state.
"""

import numpy as np
import cupy as cp

# State cell codes (kept in sync with src/gen/generate_partial_board.py)
UNKNOWN = 0
MISS = 1
HIT = 2
SUNK = 3

# Per-board (n) cache of precomputed placement candidates and 3x3 neighborhoods.
_CACHE = {}

# Above this many scattered hit components, enumerate groupings by random
# sampling instead of full Bell-curve enumeration.
_MAX_ENUM_COMPS = 10


def _to_numpy(a): #Return a numpy view/copy of a numpy or cupy array.
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _precompute(n): #Precompute all straight ship placements and 3x3 neighborhoods for an n x n board.
    if n in _CACHE:
        return _CACHE[n]

    # cands[L][o] = list of (flat_index_array, cell_set) for a length-L ship, o = 0 (horizontal) or 1 (vertical).
    cands = {}
    for L in range(1, n + 1):
        cands[L] = {0: [], 1: []}
        for o in (0, 1):
            lst = []
            if o == 0:  # horizontal
                for r in range(n):
                    for c in range(n - L + 1):
                        idxs = np.arange(c, c + L, dtype=np.int64) + r * n
                        lst.append((idxs, frozenset(idxs.tolist())))
            else:  # vertical
                for c in range(n):
                    for r in range(n - L + 1):
                        idxs = np.arange(r, r + L, dtype=np.int64) * n + c
                        lst.append((idxs, frozenset(idxs.tolist())))
            cands[L][o] = lst

    nb = [] # nb[i] = flat indices of the 3x3 neighborhood of cell i (including itself).
    for r in range(n):
        for c in range(n):
            cell = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < n and 0 <= cc < n:
                        cell.append(rr * n + cc)
            nb.append(np.asarray(cell, dtype=np.int64))

    _CACHE[n] = {"cands": cands, "nb": nb}
    return _CACHE[n]


def _hit_components(state_np):
    """Return the 8-connected components of HIT cells, each as a set of (r, c).

    Works because of the no-touch rule, where two *different* ships can never be
    orthogonally or diagonally adjacent, so a single 8-connected blob of hits
    always belongs to a single ship.
    """
    hits = set(map(tuple, np.argwhere(state_np == HIT)))
    if not hits:
        return []

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (r, c) in hits:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in hits:
                    union((r, c), nb)

    comps = {}
    for cell in hits:
        comps.setdefault(find(cell), set()).add(cell)
    return list(comps.values())


def _segment_info(cells):
    """Describe a hit cluster as a straight segment, or None if non-collinear.

    ``orientation`` is 0 (horizontal), 1 (vertical), or ``None`` for a single
    cell, whose ship could run either way.
    """
    cells = set(cells)
    if len(cells) == 1:
        return {"orientation": None, "span": 1, "cells": frozenset(cells)}
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if all(r == rows[0] for r in rows):  # horizontal
        return {"orientation": 0, "span": max(cols) - min(cols) + 1,
                "cells": frozenset(cells)}
    if all(c == cols[0] for c in cols):  # vertical
        return {"orientation": 1, "span": max(rows) - min(rows) + 1,
                "cells": frozenset(cells)}
    return None  # L-shaped / non-collinear -> cannot be one straight ship


def _set_partitions(m):
    """Yield all set partitions of {0, ..., m-1} as lists of index tuples."""
    blocks = []

    def rec(k):
        if k == m:
            yield [tuple(b) for b in blocks]
            return
        for b in blocks:
            b.append(k)
            yield from rec(k + 1)
            b.pop()
        blocks.append([k])
        yield from rec(k + 1)
        blocks.pop()

    yield from rec(0)


def _line_clear(comps, block, orientation, state_np):
    """True if all cells strictly between the block's hit components on their
    shared line are UNKNOWN (so one ship may span them)."""
    segs = []
    for i in block:
        cells = comps[i]
        if orientation == 0:  # horizontal: all on one row
            line = min(r for r, _ in cells)
            segs.append((min(c for _, c in cells), max(c for _, c in cells), line))
        else:  # vertical: all on one column
            line = min(c for _, c in cells)
            segs.append((min(r for r, _ in cells), max(r for r, _ in cells), line))
    if len({s[2] for s in segs}) != 1:
        return False
    line = segs[0][2]
    segs.sort()
    for (lo, hi, _), (blo, _, _) in zip(segs, segs[1:]):
        if orientation == 0:
            clear = all(state_np[line, c] == UNKNOWN for c in range(hi + 1, blo))
        else:
            clear = all(state_np[r, line] == UNKNOWN for r in range(hi + 1, blo))
        if not clear:
            return False
    return True


def _valid_partitions(comps, state_np, max_len, max_groups):
    """Return every valid grouping of the hit ``comps`` into straight ship
    segments: same line, only UNKNOWN cells between components, span <= max_len,
    and at most ``max_groups`` groups. Each is a list of ``_segment_info`` dicts.
    For > _MAX_ENUM_COMPS components, random groupings are sampled instead."""
    m = len(comps)
    if m == 0:
        return [[]]  # no hits: single grouping with no partially-hit ships
    valid = []

    def check(blocks):
        groups = []
        for block in blocks:
            if not block:
                continue
            cells = set()
            for i in block:
                cells |= comps[i]
            info = _segment_info(cells)
            if info is None or info["span"] > max_len:
                return
            if len(block) > 1 and not _line_clear(comps, block, info["orientation"], state_np):
                return
            groups.append(info)
        if groups and len(groups) <= max_groups:
            valid.append(groups)

    if m <= _MAX_ENUM_COMPS:
        for blocks in _set_partitions(m):
            check(blocks)
    else:
        rng = np.random.default_rng(0)
        seen = set()
        max_groups_here = min(m, max_groups)
        attempts = 0
        while len(valid) < 512 and attempts < 20000:
            attempts += 1
            nb_groups = int(rng.integers(1, max_groups_here + 1))
            blocks = [set() for _ in range(nb_groups)]
            for i in range(m):
                blocks[int(rng.integers(nb_groups))].add(i)
            key = frozenset(frozenset(b) for b in blocks if b)
            if key in seen:
                continue
            seen.add(key)
            check([tuple(b) for b in blocks])
    return valid


def _pick_covering(cands, length, orientation, cover_cells, blocked, rng):
    """Random legal placement of a length-``length`` ship (given orientation) that
    covers every cell in ``cover_cells`` (flat indices) and avoids ``blocked``.
    An ``orientation`` of ``None`` (a single hit) tries both orientations."""
    orients = (orientation,) if orientation is not None else (0, 1)
    valid = []
    for o in orients:
        for idxs, cset in cands[length][o]:
            if not cover_cells.issubset(cset):
                continue
            if blocked[idxs].any():
                continue
            valid.append(idxs)
    if not valid:
        return None
    return valid[int(rng.integers(len(valid)))]


def _pick_free(cands, length, blocked, rng):
    """Random legal placement of an un-hit length-``length`` ship avoiding ``blocked``."""
    valid = []
    for orientation in (0, 1):
        for idxs, _cset in cands[length][orientation]:
            if not blocked[idxs].any():
                valid.append(idxs)
    if not valid:
        return None
    return valid[int(rng.integers(len(valid)))]


def _mark_blocked(blocked, nb, idxs):
    """Mark a placed ship's cells and their no-touch neighborhood as blocked."""
    blocked[idxs] = True
    for i in idxs:
        blocked[nb[i]] = True


def solve(state, remaining=None, num_iterations=2000, seed=None):
    """Main function: Estimate the per-cell probability that a remaining ship occupies each cell.

    Parameters
    ----------
    state : cupy/numpy (10, 10) int array, or a ``generate_partial_state`` dict
        Cell codes: 0=unknown, 1=miss, 2=hit, 3=sunk.
    remaining : tuple/list of int
        Lengths of ships not yet sunk. Optional when ``state`` is a dict.
    num_iterations : int
        Monte-Carlo iterations to attempt; each samples a random grouping of the
        hit cells, a length assignment, and a legal placement for every remaining
        ship, and is accepted only if all ships fit (rejection sampling).
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    cupy (10, 10) float32 array: the counter of ship occupancies divided by the
    number of *successful* iterations, i.e. a Monte-Carlo estimate of
    P(cell holds a remaining-ship cell | state). Hit cells come out ~1.0;
    miss/sunk cells stay 0.0.
    """
    if isinstance(state, dict):
        if remaining is None:
            remaining = state.get("remaining")
        state = state["state"]
    if remaining is None:
        raise ValueError("`remaining` ship lengths are required")

    state_np = _to_numpy(state)
    n = state_np.shape[0]
    remaining = tuple(int(x) for x in remaining)
    out = np.zeros((n, n), dtype=np.float32)
    if len(remaining) == 0:
        return cp.asarray(out)

    pre = _precompute(n)
    cands, nb = pre["cands"], pre["nb"]
    miss_mask = (state_np == MISS).ravel()
    sunk_mask = (state_np == SUNK).ravel()

    # Groupings of the hit cells into partially-hit ships; each prepared entry
    # is a tuple of (orientation, span, flat-cell-frozenset), most-constrained
    # first. Sampling uniformly over all syntactically valid groupings keeps the
    # estimate from over-committing to any single reading of ambiguous states.
    comp_cells = _hit_components(state_np)
    groupings = _valid_partitions(comp_cells, state_np, max(remaining),
                                  len(remaining))
    if not groupings:
        raise ValueError("hit cells cannot be assigned to the remaining ships "
                         "(inconsistent state)")
    prepared = [
        tuple((info["orientation"], info["span"],
               frozenset(r * n + c for r, c in info["cells"]))
              for info in sorted(part, key=lambda info: -info["span"]))
        for part in groupings
    ]

    rng = np.random.default_rng(seed)
    counts = np.zeros(n * n, dtype=np.int64)
    accepted = 0

    for _ in range(int(num_iterations)):
        # Cells where a new ship cell may not go: misses, sunk cells, and the
        # no-touch neighborhood of sunk cells.
        blocked = miss_mask | sunk_mask
        for i in np.nonzero(sunk_mask)[0]:
            blocked[nb[i]] = True

        # --- partially-hit ships: sample a grouping, assign lengths, place ---
        part = prepared[int(rng.integers(len(prepared)))]
        avail = list(remaining)
        comp_assign = []
        ok = True
        for orientation, span, cells in part:
            choices = [L for L in avail if L >= span]
            if not choices:
                ok = False
                break
            L = choices[int(rng.integers(len(choices)))]
            avail.remove(L)
            comp_assign.append((orientation, cells, L))
        if not ok:
            continue

        placed = []
        for orientation, cells, L in comp_assign:
            idxs = _pick_covering(cands, L, orientation, cells, blocked, rng)
            if idxs is None:
                ok = False
                break
            _mark_blocked(blocked, nb, idxs)
            placed.append(idxs)
        if not ok:
            continue

        # --- other remaining ships: random order, random legal placement ---
        order = np.asarray(avail, dtype=np.int64)
        if order.size:
            rng.shuffle(order)
        for L in order.tolist():
            idxs = _pick_free(cands, int(L), blocked, rng)
            if idxs is None:
                ok = False
                break
            _mark_blocked(blocked, nb, idxs)
            placed.append(idxs)
        if not ok:
            continue

        for idxs in placed:
            counts[idxs] += 1
        accepted += 1

    if accepted:
        out = (counts.reshape(n, n) / accepted).astype(np.float32)
    return cp.asarray(out)


def solve_batch(states, remainings=None, num_iterations=2000, seed=None):
    """Solve many states; returns a cupy (N, 10, 10) float32 stack of marginals.

    ``states`` and ``remainings`` are same-length iterables. Each state may be a
    raw array (pair it with a length tuple in ``remainings``) or a full
    ``generate_partial_state`` dict (in which case ``remainings`` may be None).
    """
    if remainings is None:
        remainings = [None] * len(states)
    labels = [solve(s, r, num_iterations=num_iterations,
                    seed=(seed + i) if seed is not None else None)
              for i, (s, r) in enumerate(zip(states, remainings))]
    return cp.stack(labels)


if __name__ == "__main__":
    import os
    import sys

    # Make src/gen importable when run as a plain script.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "gen"))
    from generate_partial_board import generate_partial_state

    def truth_remaining_cells(truth, sunk):
        """Set of (r, c) cells belonging to non-sunk ships on the truth board."""
        cells = set()
        for ship_id in range(1, len(sunk) + 1):
            if sunk[ship_id - 1]:
                continue
            for r, c in np.argwhere(truth == ship_id):
                cells.add((int(r), int(c)))
        return cells

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1234
    game = generate_partial_state(seed, policy="expand")
    state = cp.asnumpy(game["state"])
    remaining = game["remaining"]

    label = cp.asnumpy(solve(state, remaining, num_iterations=5000, seed=0))
    truth_set = truth_remaining_cells(cp.asnumpy(game["truth"]), game["sunk"])

    print(f"seed={seed}  remaining={remaining}  hits={int((state == HIT).sum())}")
    print("label (per-cell P of remaining ship):")
    print(np.round(label, 2))
    print("truth remaining-ship cells:", sorted(truth_set))

    # --- sanity checks ---
    def cell_vals(cellset):
        """Label values at the given (r, c) cells."""
        if not cellset:
            return np.array([])
        rows = np.array([r for r, _ in cellset])
        cols = np.array([c for _, c in cellset])
        return label[rows, cols]

    hit_cells = set(map(tuple, np.argwhere(state == HIT)))
    miss_cells = set(map(tuple, np.argwhere(state == MISS)))
    sunk_cells = set(map(tuple, np.argwhere(state == SUNK)))
    assert np.all(cell_vals(hit_cells) > 0.99), "hit cells should be ~1.0"
    assert np.all(cell_vals(miss_cells) < 0.01), "miss cells should be ~0.0"
    assert np.all(cell_vals(sunk_cells) < 0.01), "sunk cells should be ~0.0"

    # determinism: same seed -> identical label
    assert np.allclose(cp.asnumpy(solve(state, remaining, 5000, seed=0)),
                       cp.asnumpy(solve(state, remaining, 5000, seed=0)))

    # top-1 cell should be a true remaining-ship cell (soft, printed only)
    top = np.unravel_index(np.argmax(label), label.shape)
    print(f"top-1 cell {tuple(top)} in truth-remaining set: {top in truth_set}")

    # batch API
    games = [generate_partial_state(seed + k, policy="expand") for k in range(3)]
    batch = solve_batch([g["state"] for g in games],
                        [g["remaining"] for g in games],
                        num_iterations=2000, seed=7)
    print(f"solve_batch shape: {tuple(batch.shape)} (expect ({len(games)}, 10, 10))")
    print("OK")
