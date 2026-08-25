# Board generator, main function returns a full board (5/4/3/3/2 length boats that are not touching) in a deterministic (seeded) manor.

import numpy as np
import cupy as cp #cupy for GPU-accelerated numpy

def generate_board(input_seed):
    np.random.seed (seed = input_seed)
    