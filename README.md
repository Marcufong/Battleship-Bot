# battleship-bot

A battleship bot that is comprised of a CNN that is trained to imitate an exact probabilistic solver, then RL-tuned for **full-game** performance. The post RL-tuning performance is then evaluated at different CNN parameter counts (`1B → 100M → 10M → 1M → 100K → 10K`) as a **pruning study**.

## Pipeline overview

```
board gen (fast_generate.py)
   ↓  random legal boards (ships 5/4/3/3/2, no-touch)
state gen  →  partial-game states (miss/hit/sunk/remaining info)
   ↓
exact/MC probabilistic solver  →  per-cell hit-probability map (the "teacher" label)
   ↓
supervised CNN training  →  "large" model (intentionally very overparameterized)
   ↓
iterative magnitude pruning + fine-tune  →  models at each 10× size target
   ↓
PPO RL-tuning  →  optimizes moves for entire games, rather than single turns
   ↓
eval + writeup  →  fidelity & full-game curves vs params; pruning-vs-scratch comparison
```

## Key points/ideas
- **Single-turn vs full-game**: myopic imitation (fire at highest-probability cell) isn't optimal over a full game,RL-tuning aims to fix this.
- **Pruning question**: does a 1B model pruned down to 1M beat a 1M model trained from scratch? (lottery-ticket style) — and where does fidelity cliff?
- **Effective params** = non-zero weight count (unstructured pruning); structured pruning is a bonus axis.

## Layout
`src/gen/`, `src/solver/`, `src/model/`, `src/prune/`, `src/rl/`, `src/eval/`; `data/`, `checkpoints/`, `results/`, `plots/`.

See `plan.md` for full plan.
