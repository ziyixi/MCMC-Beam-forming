""" 
optimization.py

collection of optimization functions
"""


from typing import Optional

import numpy as np
from mpi4py import MPI
from numba import njit
from numpy.typing import NDArray

from bbf.beamforming.bf import FreqBF


def brute_force_search_mpi(
    bf: FreqBF,
    phis: NDArray[np.float64],
    thetas: NDArray[np.float64],
    vs: NDArray[np.float64],
    comm: Optional[MPI.Comm] = None,
) -> NDArray[np.float64]:
    all_cases = []
    all_cases_indices = []
    for iphi, phi in enumerate(phis):
        for itheta, theta in enumerate(thetas):
            for iv, v in enumerate(vs):
                all_cases.append((phi, theta, v))
                all_cases_indices.append((iphi, itheta, iv))
    all_cases_this_rank = np.array_split(all_cases, comm.Get_size())[comm.Get_rank()]
    res = np.zeros(len(all_cases_this_rank))
    for idx, case in enumerate(all_cases_this_rank):
        res[idx] = bf.opt_func(case)
    # gather to rank 0
    res_gathered = comm.gather(res, root=0)

    if comm.Get_rank() == 0:
        res_all = np.zeros((len(phis), len(thetas), len(vs)))
        for rank in range(comm.Get_size()):
            all_cases_indices_current_rank = np.array_split(
                all_cases_indices, comm.Get_size()
            )[rank]
            res_current_rank = res_gathered[rank]
            for idx, (iphi, itheta, iv) in enumerate(all_cases_indices_current_rank):
                res_all[iphi, itheta, iv] = res_current_rank[idx]
    else:
        res_all = None
    # broadcast to all ranks
    res_all = comm.bcast(res_all, root=0)

    return res_all


@njit()
def brute_force_search(
    bf: FreqBF,
    phis: NDArray[np.float64],
    thetas: NDArray[np.float64],
    vs: NDArray[np.float64],
) -> NDArray[np.float64]:
    res = np.zeros((len(phis), len(thetas), len(vs)))
    for i, phi in enumerate(phis):
        for j, theta in enumerate(thetas):
            for k, v in enumerate(vs):
                res[i, j, k] = bf.opt_func((phi, theta, v))
    return res
