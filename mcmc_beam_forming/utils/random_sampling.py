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
    seen_samples = set()
    np.random.seed(RANDOM_SAMPLING_SEED)

    for key1 in sorted(grids_total.keys()):
        for key2 in sorted(grids_total[key1].keys()):
            data = grids_total[key1][key2]
            array_length = len(data)

            subsamples = generate_subsampling_indices(
                array_length, num_samples=RANDOM_SAMPLE_TIMES
            )

            for idx_set in subsamples:
                sample = tuple(sorted(data[i] for i in idx_set))
                seen_samples.add(sample)

    seen_samples_list = sorted(list(seen_samples))
    return seen_samples_list
