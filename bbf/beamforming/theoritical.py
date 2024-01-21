from pathlib import Path

import numpy as np
import pandas as pd
import pymap3d as pm
from numpy.typing import NDArray
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
from scipy import interpolate

model = TauPyModel(model="iasp91")


def get_theoritical_azi_v_takeoff(
    coordinates_lld: NDArray[np.float64], station_lld: NDArray[np.float64]
):
    slon, slat = station_lld
    ref_lat = np.mean(coordinates_lld[:, 0])
    ref_lon = np.mean(coordinates_lld[:, 1])
    ref_dep = np.mean(coordinates_lld[:, 2])

    e, n, u = pm.geodetic2enu(slat, slon, 0, ref_lat, ref_lon, -ref_dep * 1000)
    azi = np.rad2deg(np.arctan2(n, e)) + 180

    dir_path = Path(__file__).parent
    csv_path = dir_path / "IASP91.csv"
    df_velocity = pd.read_csv(csv_path, names=["depth", "r", "vp", "vs"])
    f_vp = interpolate.interp1d(df_velocity.depth, df_velocity.vp)
    f_vs = interpolate.interp1d(df_velocity.depth, df_velocity.vs)
    target_vp, target_vs = f_vp(ref_dep), f_vs(ref_dep)

    arrival = model.get_travel_times(
        source_depth_in_km=ref_dep,
        distance_in_degree=locations2degrees(ref_lat, ref_lon, slat, slon),
        phase_list=["P", "p"],
    )[0]
    takeoff = 90 - arrival.takeoff_angle

    return azi, target_vp, target_vs, takeoff
