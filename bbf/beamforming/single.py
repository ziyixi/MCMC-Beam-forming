from typing import List, Optional

import numpy as np
import pandas as pd
from mpi4py import MPI

from bbf.beamforming.bf import FreqBF
from bbf.beamforming.optimization import brute_force_search, brute_force_search_mpi
from bbf.beamforming.theoritical import get_theoritical_azi_v_takeoff
from bbf.beamforming.utils import generate_waveforms, read_time_info
from bbf.setting import PARSE_OMEGA_RATIO, WAVEFORM_DELTA


def bf_single(
    info_df: pd.DataFrame,
    resolution: List[float] = [2, 2, 0.1],
    station_lld: Optional[List[float]] = None,
    fix_theta_in_search_value: Optional[float] = None,
    phase_key: str = "P",
    comm: Optional[MPI.Comm] = None,
):
    # * generate waveforms based on the info table
    arrival_times, coordinates = read_time_info(info_df, phase_key=phase_key)
    waveforms = generate_waveforms(arrival_times)

    # now we construct the opt search class
    # waves and coors should be wraped into the numpy array
    m = len(waveforms)
    n = len(waveforms[list(waveforms.keys())[0]].data)
    waves = np.zeros((m, n), dtype=np.float64)
    coors = np.zeros((m, 3), dtype=np.float64)
    for idx, k in enumerate(waveforms):
        waves[idx, :] = waveforms[k].data[:n]
        coors[idx, :] = coordinates[k]

    # do optimization
    bf = FreqBF(waves, coors, WAVEFORM_DELTA, PARSE_OMEGA_RATIO)
    rrange = {
        "phi": np.arange(-90, 90, resolution[0]),
        "theta": np.arange(0, 360, resolution[1])
        if fix_theta_in_search_value is None
        else np.array([fix_theta_in_search_value]),
        "v": np.arange(5.5, 11.5, resolution[2]),
    }

    # do optimization
    if comm is not None:
        brute_force_res = brute_force_search_mpi(
            bf, rrange["phi"], rrange["theta"], rrange["v"], comm=comm
        )
    else:
        brute_force_res = brute_force_search(
            bf, rrange["phi"], rrange["theta"], rrange["v"]
        )
    opt_res = np.unravel_index(brute_force_res.argmax(), brute_force_res.shape)

    # collect final results
    res = {
        "phi_opt": rrange["phi"][opt_res[0]],
        "theta_opt": rrange["theta"][opt_res[1]],
        "v_opt": rrange["v"][opt_res[2]],
        "amplitude_opt": brute_force_res[opt_res[0], opt_res[1], opt_res[2]],
    }

    azi, target_vp, target_vs, takeoff = get_theoritical_azi_v_takeoff(
        coors, station_lld
    )
    res.update(
        {
            "theoritical_azi": azi,
            "theoritical_vp": target_vp,
            "theoritical_vs": target_vs,
            "theoritical_takeoff": takeoff,
        }
    )

    return res
