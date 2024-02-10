from typing import Tuple

import numpy as np
import pandas as pd


def get_grids(
    lon0: float,
    lon1: float,
    lat0: float,
    lat1: float,
    lon_step: float,
    lat_step: float,
    depth_step: float,
    horizontal_size: float,
    vertical_size: float,
    info: pd.DataFrame,
    minimum_number_of_ps_events_in_box: int,
) -> Tuple[dict, pd.DataFrame]:
    # * 1. Filter the arrival info
    arrival_info = info.copy()
    arrival_info["ORIGIN_TIME"] = pd.to_datetime(arrival_info["ORIGIN_TIME"])
    arrival_info["PTIME"] = pd.to_datetime(arrival_info["PTIME"])
    arrival_info["PSTIME"] = pd.to_datetime(arrival_info["PSTIME"])
    # map info["ELON"] to [0,360]
    arrival_info["ELON"] = arrival_info["ELON"].apply(lambda x: x + 360 if x < 0 else x)

    # remove unnecessary columns
    if "STIME" in arrival_info.columns:
        arrival_info = arrival_info.drop(columns=["STIME"])
    # remove rows where PSTIME is NaT or PTIME is NaT
    arrival_info = arrival_info.dropna(subset=["PSTIME", "PTIME"])
    # reindex
    arrival_info = arrival_info.reset_index(drop=True)
    arrival_info["INDEX"] = arrival_info.index

    all_stations = set(arrival_info["STATION"])

    grids_raw = {}

    # * 2. Get the grids that covers all stations

    # Function to check if one set of indexes completely covers another
    def is_covered(set_a, set_b):
        return set_a.issuperset(set_b)

    grids_index = 0
    for center_lon in np.arange(
        lon0 + horizontal_size / 2, lon1 - horizontal_size / 2 + lon_step, lon_step
    ):
        for center_lat in np.arange(
            lat0 + horizontal_size / 2, lat1 - horizontal_size / 2 + lat_step, lat_step
        ):
            for center_dep in np.arange(
                vertical_size / 2, 700 - vertical_size / 2 + depth_step, depth_step
            ):
                box_lon = [
                    center_lon - horizontal_size / 2,
                    center_lon + horizontal_size / 2,
                ]
                box_lat = [
                    center_lat - horizontal_size / 2,
                    center_lat + horizontal_size / 2,
                ]
                box_dep = [
                    center_dep - vertical_size / 2,
                    center_dep + vertical_size / 2,
                ]
                filtered_info = arrival_info[
                    (arrival_info["ELON"] >= box_lon[0])
                    & (arrival_info["ELON"] <= box_lon[1])
                    & (arrival_info["ELAT"] >= box_lat[0])
                    & (arrival_info["ELAT"] <= box_lat[1])
                    & (arrival_info["EDEP"] >= box_dep[0])
                    & (arrival_info["EDEP"] <= box_dep[1])
                ]
                if len(filtered_info) >= minimum_number_of_ps_events_in_box:
                    indexes = sorted(filtered_info["INDEX"].tolist())
                    indexes_set = set(indexes)

                    is_new_grid_covered = False
                    grids_to_remove = []
                    for existing_indexes in grids_raw.keys():
                        if is_covered(set(existing_indexes), indexes_set):
                            is_new_grid_covered = True
                            break
                        if is_covered(indexes_set, set(existing_indexes)):
                            grids_to_remove.append(existing_indexes)

                    for key in grids_to_remove:
                        del grids_raw[key]

                    if not is_new_grid_covered:
                        grids_raw[tuple(indexes)] = (
                            center_lon,
                            center_lat,
                            center_dep,
                            grids_index,
                        )
                        grids_index += 1

    # * 3. Filter the grids for each station
    # for each grid, filter info for each station, and filter info in a new dict only if occurance > 30
    grids = {}
    for index_this_grid, positions in grids_raw.items():
        info_this_grid = arrival_info.loc[list(index_this_grid)]
        for sta in all_stations:
            info_this_sta = info_this_grid[info_this_grid["STATION"] == sta]
            if len(info_this_sta) >= minimum_number_of_ps_events_in_box:
                if sta not in grids:
                    grids[sta] = {}
                index_this_grid_this_sta = sorted(info_this_sta["INDEX"].tolist())
                grids[sta][positions] = index_this_grid_this_sta

    return grids, arrival_info
