import numpy as np
from numba import njit

from mcmc_beam_forming.core.bf import FreqBF


@njit()
def brute_force_search(
    bf: FreqBF,
    phis,
    thetas,
    vs,
):
    # Initialize max value and corresponding parameters
    max_value = -np.inf
    max_phi = max_theta = max_v = np.nan

    # Iterate through all combinations of phi, theta, and v
    for phi in phis:
        for theta in thetas:
            for v in vs:
                current_value = bf.opt_func((phi, theta, v))
                if current_value > max_value:
                    max_value = current_value
                    max_phi = phi
                    max_theta = theta
                    max_v = v

    return max_phi, max_theta, max_v, max_value
