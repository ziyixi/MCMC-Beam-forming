from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
from mpi4py import MPI

from bbf.beamforming.single import bf_single
from bbf.setting import BOOTSTRAP_SAMPLE_RATIO, BOOTSTRAP_SEED, BOOTSTRAP_TIMES


def bootstrap(
    info_df: pd.DataFrame,
    resolution: List[float] = [2, 2, 0.1],
    station_lld: Optional[List[float]] = None,
    fix_theta_in_search_value: Optional[float] = None,
    theoritical_vp: Optional[float] = None,
    comm: Optional[MPI.Comm] = None,
):
    np.random.seed(BOOTSTRAP_SEED + comm.Get_rank())
    misfit_collection_this_process = defaultdict(list)

    for _ in range(BOOTSTRAP_TIMES // comm.Get_size()):
        info_df_this_iter = info_df.sample(frac=BOOTSTRAP_SAMPLE_RATIO)
        res = bf_single(
            info_df_this_iter,
            resolution=resolution,
            station_lld=station_lld,
            fix_theta_in_search_value=fix_theta_in_search_value,
            phase_key="PS",
            comm=None,  # we use None here to use single process for this iteration
        )
        misfit = 1 - np.exp(-0.5 * (res["v_opt"] - theoritical_vp) ** 2)
        # misfit = (res["v_opt"] - theoritical_vp) ** 2

        for _, row in info_df_this_iter.iterrows():
            misfit_collection_this_process[row["EVENT_ID"]].append(misfit)

    # gather the misfit collection to all processes
    misfit_collection_collected = comm.allgather(misfit_collection_this_process)
    misfit_collection = defaultdict(list)
    for each in misfit_collection_collected:
        for k, v in each.items():
            misfit_collection[k].extend(v)

    misfit_average_collection = {}
    for k, v in misfit_collection.items():
        if len(v) == 0:
            misfit_average_collection[k] = np.nan
        else:
            misfit_average_collection[k] = np.mean(v)

    return misfit_average_collection


def analyze_bootstrap_results(
    info_df: pd.DataFrame,
    bootstrap_results: dict,
    misfit_threshold_list: List[float],
    resolution: List[float] = [2, 2, 0.1],
    station_lld: Optional[List[float]] = None,
    comm: Optional[MPI.Comm] = None,
):
    result_for_each_threshold = []
    for threshold in misfit_threshold_list:
        considered_misfit = np.percentile(
            list(bootstrap_results.values()), threshold * 100
        )
        considered_events = [
            key
            for key, value in bootstrap_results.items()
            if value <= considered_misfit
        ]
        info_df_considered = info_df[info_df["EVENT_ID"].isin(considered_events)]

        # run ps wave beamforming
        res = bf_single(
            info_df_considered,
            resolution=resolution,
            station_lld=station_lld,
            phase_key="PS",
            comm=comm,
        )
        result_for_each_threshold.append(res)
    return result_for_each_threshold
