# Battleship Pruning × RL-Reward Study

## Goal
Map how pruning a trained Battleship policy down to every smaller size affects **post-RL full-game performance** (average turns to sink all enemy ships), under two reward formulations — *flat* (`r = -1/turn`) and *aggressive* (convex cost in game length). Primary artifact: a 2D heatmap (original size × final size) per reward shape, with the no-pruning diagonal as control.

## Design decisions (agreed)
- **Fidelity metric dropped** (KL is ill-defined here — the solver's output is a per-cell marginal, not a distribution over cells). The outcome metric *mean turns post-RL* is the only headline signal. Optional cheap diagnostics (top-1 cell agreement, rank correlation) may be logged later for interpretation, but are NOT part of the artifact.
- **Aggressive reward = terminal penalty** `-λ·T²` (and optionally `-λ·exp(T)`), added at game end, on top of `r = -1/turn`. This is a function of the outcome `T` (game length), not the trajectory index — avoids the discounting conflict and exploding late-game rewards of a literal exponential-in-t cost.
- **Order: prune → RL-tune → eval.** Imitation CNN is the substrate being pruned (warm-start for RL); we do not measure its fidelity.
- **Full sweep sizes:** `100M → 10M → 1M → 100K → 10K → 1K → 100` (7 sizes). Grid is upper-triangular (pruning only shrinks): 7 diagonal cells + 21 pruning pairs = **28 cells**.
- **Pilot first:** `10K, 100K, 1M` × flat + aggressive, to validate pipeline + reward comparison before committing to the full grid.
- **Eval:** thousands of simulated games per policy; report **mean turns + tail (P90/P95)** — the tail is where aggressiveness should show its effect.

## Pipeline

### Phase 0 — Teacher & imitation CNNs
- `src/gen/`: random legal boards (ships 5/4/3/3/2, no-touch), deterministic seeds; partial-game state generator (board → known misses/hits → training state).
- `src/solver/`: exact probabilistic targeting solver → imitation targets.
- `src/model/`: train an imitation CNN at each sweep size (warm-start substrate for pruning; no fidelity measurement — downstream turns are the signal).

### Phase 1 — Pruning grid
- Iterative magnitude pruning (sparsity schedule): each original size → every final size ≤ original. Diagonal = no-op control.
- Pilot grid: `1M→100K`, `1M→10K`, `100K→10K` + 3 diagonal cells.

### Phase 2 — RL tuning
- PPO on each (original, final) × reward shape; 2–3 seeds per config.
- Reward shapes:
  - flat: `r = -1/turn` (total = `-T`).
  - aggressive: `r = -1/turn` + terminal `-λ·T²` (and later a `-λ·exp(T)` variant). λ tuned on the pilot.

### Phase 3 — Eval & plots
- Eval harness: simulate thousands of games/policy; metrics = mean turns + P90/P95.
- 2D heatmap per reward shape: x = final size, y = original size, value = mean turns; diagonal = no-pruning control (isolates pruning effect off-diagonal, capacity effect along-diagonal).
- Reward comparison: flat vs aggressive on mean + tail.

### Phase 4 — Writeup
- Heatmaps, pruning-effect curves, reward-shape comparison, tail analysis.

## Compute notes
- Pilot: 3 sizes → 6 cells × 2 rewards = 12 RL runs (× seeds). Validate before full grid.
- Full grid: 7 sizes → 28 cells × 2–3 rewards. Battleship episodes are short and CNNs ≤ 100M ⇒ tractable on the Spark.

## Open items (confirm before full grid)
- λ for the aggressive terminal penalty (tune on pilot).
- Reward shapes to include in the full grid (flat, `-λT²`, ± `-λexp(T)`).
- Seeds / RL runs per cell.
- Sweep sizes final: `100M…100` confirmed; pilot `10K/100K/1M` confirmed.
