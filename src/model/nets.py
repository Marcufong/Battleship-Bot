"""
Parameter-targeted CNN for the Battleship imitation model (M0).

Architecture (width-only scaling, depth/stride are fixed and the
channel table below is what future width-surgery prunes layer-by-layer):

    9 input planes (each 10x10)
        ch 0-3 : one-hot of the state cell code (0=unknown, 1=miss, 2=hit, 3=sunk)
        ch 4-8 : 1.0 where the corresponding fleet ship (1-based id) is NOT yet sunk
        |
        v
    Conv2d(9, W, 3x3, pad=1) -> ReLU
    Conv2d(W, W, 3x3, pad=1) -> ReLU
    Conv2d(W, W, 3x3, pad=1) -> ReLU
        |
        +-- policy head: Conv2d(W, 1, 1x1) -> per-cell logits (B, 10, 10)
        |                 (trained in M0 to imitate the solver's per-cell marginal)
        +-- value head : spatial mean -> Linear(W, 1) -> scalar (B, 1)
                          (initialized only here; PPO in M2 will train it)

Trainable parameter count for trunk width W:
    T(W) = (81W + W) + 2 * (9W^2 + W) + (W + 1) + (W + 1)
         = 18*W^2 + 86*W + 2
"""

import torch
import torch.nn as nn

BOARD_SIZE = 10
NUM_STATE_CODES = 4          # unknown / miss / hit / sunk
FLEET = (5, 4, 3, 3, 2)      # standard fleet (mirrors src/gen/generate_board.py)
NUM_SHIPS = len(FLEET)
INPUT_CHANNELS = NUM_STATE_CODES + NUM_SHIPS  # 9 planes, all 10x10

TRUNK_LAYERS = 3             # fixed depth
KERNEL_SIZE = 3              # fixed kernel; stride 1 + zero padding keeps 10x10


def count_params(model: nn.Module) -> int:
    """Count trainable (dense) parameters -- the number pruning targets."""
    return sum(p.numel() for p in model.parameters())


class BattleshipCNN(nn.Module):
    """Conv trunk (width W) + policy head (per-cell logits) + value head (scalar)."""

    def __init__(self, in_channels: int = INPUT_CHANNELS, width: int = 32):
        super().__init__()
        trunk = []
        in_c = in_channels
        for _ in range(TRUNK_LAYERS):
            trunk.append(nn.Conv2d(in_c, width, kernel_size=KERNEL_SIZE,
                                   padding=KERNEL_SIZE // 2))
            trunk.append(nn.ReLU())
            in_c = width
        self.trunk = nn.Sequential(*trunk)
        self.policy_head = nn.Conv2d(width, 1, kernel_size=1)   # -> (B, 10, 10)
        self.value_head = nn.Linear(width, 1)                   # -> (B, 1)

    def forward(self, x: torch.Tensor):
        """x: (B, 9, 10, 10). Returns (policy (B, 10, 10), value (B, 1))."""
        h = self.trunk(x)                          # (B, W, 10, 10)
        policy = self.policy_head(h).squeeze(1)    # (B, 10, 10)
        value = self.value_head(h.mean(dim=(2, 3)))  # (B, 1)
        return policy, value


# param target -> trunk width W.  T(W) = 18*W^2 + 86*W + 2 (see module docstring).
# Tiny targets sit a bit off the round number because of the fixed linear
# overhead (86W + 2); PARAM_TOL absorbs that.
CHANNELS_BY_TARGET = {
    100: 1,
    1_000: 5,
    10_000: 21,
    100_000: 72,
    1_000_000: 233,
    10_000_000: 743,
    100_000_000: 2355,
}
PARAM_TOL = 0.15  # assert abs(count/target - 1) < tol at construction


def make_net(param_target: int) -> nn.Module:
    """Build a BattleshipCNN whose param count hits param_target within PARAM_TOL."""
    try:
        width = CHANNELS_BY_TARGET[param_target]
    except KeyError:
        raise ValueError(
            f"no recorded channel config for param_target={param_target}; "
            f"known targets: {sorted(CHANNELS_BY_TARGET)}"
        ) from None
    net = BattleshipCNN(width=width)
    count = count_params(net)
    assert abs(count / param_target - 1.0) < PARAM_TOL, (
        f"param count {count} deviates from target {param_target} by "
        f"{abs(count / param_target - 1.0):.1%} (> {PARAM_TOL:.0%})"
    )
    return net


if __name__ == "__main__":
    # Construction check (param-count assert) + tiny forward pass for every size.
    x = torch.randn(2, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    for target in sorted(CHANNELS_BY_TARGET):
        net = make_net(target)
        policy, value = net(x)
        assert policy.shape == (2, BOARD_SIZE, BOARD_SIZE), policy.shape
        assert value.shape == (2, 1), value.shape
        count = count_params(net)
        print(f"target={target:>12,}  actual={count:>12,}  "
              f"width={CHANNELS_BY_TARGET[target]:>5}  delta={count / target - 1.0:+.3%}  OK")
    print("nets.py self-check OK")
