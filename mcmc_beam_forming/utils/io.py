from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd
from obspy import Trace, UTCDateTime

from mcmc_beam_forming.utils.setting import (
    LABEL_WIDTH,
    WAVEFORM_CLUSTER_LEFT_BUFFER,
    WAVEFORM_CLUSTER_RIGHT_BUFFER,
    WAVEFORM_DELTA,
)


def read_time_info(
    info_df: pd.DataFrame,
    phase_key: str = "P",
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """read the waveform phase arrival time

    Args:
        time_pd_file (str): the pd data file containing the phase arrival information
        phase_key (str): the phase key corresponding the pd file column

    Returns:
        Tuple[Dict[str, float], Dict[str, np.ndarray]]: the phase arrival info
    """

    def prepare_df(row):
        return pd.Series(
            [
                f"{row['NETWORK']}.{row['STATION']}.{row['EVENT_ID']}",
                (row["PTIME"] - row["ORIGIN_TIME"]).total_seconds(),
                (row["PSTIME"] - row["ORIGIN_TIME"]).total_seconds(),
                row["ELAT"],
                row["ELON"],
                row["EDEP"],
            ],
            index=["key", "arrival_time_p", "arrival_time_ps", "lat", "lon", "dep"],
        )

    data = info_df.apply(prepare_df, axis=1)

    arrival_times_p = {}
    arrival_times_ps = {}
    coordinates = {}
    for i in range(len(data)):
        row = data.iloc[i]
        key = row["key"]

        arrival_times_p[key] = row["arrival_time_p"]
        arrival_times_ps[key] = row["arrival_time_ps"]

        lat = float(row["lat"])
        lon = float(row["lon"])
        dep = float(row["dep"])

        coordinates[key] = np.array([lat, lon, dep])

    if phase_key == "P":
        return arrival_times_p, coordinates
    elif phase_key == "PS":
        return arrival_times_ps, coordinates


def generate_waveforms(arrival_times: Dict[str, float]) -> Dict[str, Trace]:
    # get the min and max time for all keys in arrival_times
    min_time = min(arrival_times.values())
    max_time = max(arrival_times.values())
    start_time = min_time - WAVEFORM_CLUSTER_LEFT_BUFFER
    end_time = max_time + WAVEFORM_CLUSTER_RIGHT_BUFFER
    ref_utc_time = UTCDateTime(2024, 1, 1, 0, 0, 0)
    start_time = ref_utc_time + start_time
    end_time = ref_utc_time + end_time

    # generate a gaussian waveform as a template
    label_window = np.exp(
        -((np.arange(-LABEL_WIDTH, LABEL_WIDTH + 1)) ** 2)
        / (2 * (LABEL_WIDTH / 6) ** 2)
    )

    # generate the waveforms
    waveforms = {}
    for key, arrival_time in arrival_times.items():
        arrival_time = ref_utc_time + arrival_time
        # generate the waveform
        waveform = Trace()
        waveform.stats.delta = WAVEFORM_DELTA
        waveform.stats.starttime = start_time
        waveform.data = np.zeros(int((end_time - start_time) / WAVEFORM_DELTA))

        start_pos = int((arrival_time - start_time) / WAVEFORM_DELTA)
        end_pos = start_pos + label_window.shape[0]
        waveform.data[start_pos:end_pos] = label_window
        waveforms[key] = waveform

    return waveforms


def save_dict_to_hdf5(dic, hf, path=""):
    """Recursively save dictionary to hdf5 file."""
    for key, item in dic.items():
        key_path = f"{path}/{key}" if path else key  # Create a nested path in HDF5 file

        # Check if the dataset already exists, and if so, delete it
        if key_path in hf:
            del hf[key_path]

        if isinstance(item, pd.DataFrame):
            # Convert DataFrame to CSV and store as a string
            csv_string = item.to_csv(index=True)
            hf.create_dataset(key_path, data=csv_string)
        elif isinstance(item, dict):
            # If a sub-dictionary, create a group (if it doesn't exist)
            if key_path not in hf:
                hf.create_group(key_path)
            # Recursively save dictionary
            save_dict_to_hdf5(item, hf, key_path)
        else:
            # Directly save other data types
            hf[key_path] = item


def write_to_hdf5(data: dict, output_file: str, key: str):
    with h5py.File(output_file, "a") as f:
        save_dict_to_hdf5(data, f, key)
