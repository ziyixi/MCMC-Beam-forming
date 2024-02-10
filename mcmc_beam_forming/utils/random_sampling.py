import numpy as np
from numba import njit

from mcmc_beam_forming.utils.setting import (
    RANDOM_SAMPLE_TIMES,
    RANDOM_SAMPLING_RATIO,
    RANDOM_SAMPLING_SEED,
)


@njit
def generate_subsampling_indices(array_length, num_samples):
    """
    Generate random indices for subsampling using NumPy, in a batched manner.
    Optimized with Numba.
    """
    subsample_size = int(array_length * RANDOM_SAMPLING_RATIO)
    indices = np.empty((num_samples, subsample_size), dtype=np.int64)
    for i in range(num_samples):
        indices[i] = np.random.choice(array_length, size=subsample_size, replace=False)
    return indices


def unique_subsamples(grids_total: dict) -> iter:
    """
    Generate unique subsamples from grids_total using NumPy for random index generation.
    Optimized with Numba.
    """
    seen_samples = []
    np.random.seed(RANDOM_SAMPLING_SEED)

    for sta in sorted(grids_total.keys()):
        for position in sorted(grids_total[sta].keys()):
            data = grids_total[sta][position]
            array_length = len(data)

            subsamples = generate_subsampling_indices(
                array_length, num_samples=RANDOM_SAMPLE_TIMES
            )

            for idx_set in subsamples:
                sample = tuple(sorted(data[i] for i in idx_set) + [sta, position])
                seen_samples.append(sample)

    seen_samples.sort()
    return seen_samples
