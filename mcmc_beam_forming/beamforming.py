from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pandas as pd
from loguru import logger
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from mcmc_beam_forming.core.bf import FreqBF
from mcmc_beam_forming.utils.get_grids import get_grids
from mcmc_beam_forming.utils.io import (
    generate_waveforms,
    read_time_info,
    write_to_hdf5,
)
from mcmc_beam_forming.utils.optim import brute_force_search
from mcmc_beam_forming.utils.random_sampling import unique_subsamples
from mcmc_beam_forming.utils.theoritical import get_theoritical_azi_v_takeoff

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def bf_wrapper(parameter: Tuple[pd.DataFrame, Path, int]):
    df, output_directory_path, itask = parameter
    # * generate waveforms based on the info table
    station_lld = [df["SLON"].iloc[0], df["SLAT"].iloc[0]]
    arrival_times, coordinates = read_time_info(df, phase_key="PS")
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

    # * construct the class and do optimization
    bf = FreqBF(waves, coors)
    rrange = {
        "phi": np.arange(-90, 90, 2),
        "theta": np.arange(0, 360, 2),
        "v": np.arange(5.5, 11.5, 0.1),
    }

    takeoff_opt, azi_opt, v_opt, amplitude_opt = brute_force_search(
        bf, rrange["phi"], rrange["theta"], rrange["v"]
    )
    azi_theoritical, v_theoritical, takeoff_theoritical = get_theoritical_azi_v_takeoff(
        coors, station_lld
    )
    res = {
        "takeoff_opt": takeoff_opt,
        "azi_opt": azi_opt,
        "v_opt": v_opt,
        "amplitude_opt": amplitude_opt,
        "azi_theoritical": azi_theoritical,
        "v_theoritical": v_theoritical.item(),
        "takeoff_theoritical": takeoff_theoritical,
    }
    # save key is the sorted index of the dataframe
    key = tuple(sorted(df["INDEX"].tolist()))
    output_file_path = output_directory_path / f"{rank}.h5"
    write_to_hdf5(res, str(output_file_path), key)
    logger.info(f"Process [{rank}/{size}] Finished Task {itask} with length {len(df)}")


@click.command()
@click.option(
    "--coordinates",
    default="176,188,-26,24",
    help="Coordinates to search for, in the format of 'lon0,lon1,lat0,lat1', 0<=lon<=360, -90<=lat<=90",
)
@click.option(
    "--search_step_length",
    default="0.25,0.25,25",
    help="Step length for searching, in the format of 'lon,lat,depth'",
)
@click.option(
    "--box_size",
    default="1,100",
    help="The size of the box, in the format of 'horizontal_size,vertical_size', horizontal size is in degree, vertical size is in km",
)
@click.option(
    "--arrival_info_file",
    help="The csv file name of the arrival information, it must contain columns including 'EVENT_ID,ORIGIN_TIME,STATION,NETWORK,ELON,ELAT,EDEP,PTIME,PSTIME,SLON,SLAT'",
    required=True,
)
@click.option(
    "--minumum_number_of_ps_events_in_box",
    default=20,
    help="The minimum number of ps events in the box",
    type=int,
)
@click.option(
    "--output_directory",
    default="mcmcbf_output",
    help="The output hdf5 file name, containing the results of the beamforming and bootstrap",
)
def main(
    coordinates: str,
    search_step_length: str,
    box_size: str,
    arrival_info_file: str,
    minumum_number_of_ps_events_in_box: int,
    output_directory: str,
):
    # * parse the parameters
    lon0, lon1, lat0, lat1 = [float(x) for x in coordinates.split(",")]
    lon_step, lat_step, depth_step = [float(x) for x in search_step_length.split(",")]
    horizontal_size, vertical_size = [float(x) for x in box_size.split(",")]
    arrival_info_raw = pd.read_csv(arrival_info_file)
    output_directory_path = Path(output_directory)
    output_directory_path.mkdir(parents=True, exist_ok=True)

    # * get all possible indexes of arrival_info to do beamforming
    grids_total, arrival_info = get_grids(
        lon0,
        lon1,
        lat0,
        lat1,
        lon_step,
        lat_step,
        depth_step,
        horizontal_size,
        vertical_size,
        arrival_info_raw,
        minumum_number_of_ps_events_in_box,
    )
    logger.info(
        f"Total number of grids: {sum([len(grids_total[key]) for key in grids_total])}"
    )
    # save arrival_info to output_directory_path/arrival_info.csv
    arrival_info.to_csv(output_directory_path / "arrival_info.csv")

    subsamples_list = unique_subsamples(grids_total)
    logger.info(
        f"Total number of subsamples: {len(subsamples_list)}, start to beamforming"
    )

    def yield_arrival_info_for_subsample(subsample_lst):
        for itask, subsample_item in enumerate(subsample_lst):
            subsample_item = list(subsample_item)
            yield (
                arrival_info.loc[subsample_item],
                output_directory_path,
                itask,
            )

    with MPIPoolExecutor() as executor:
        executor.map(
            bf_wrapper, yield_arrival_info_for_subsample(subsamples_list[:100])
        )

    logger.info("Beamforming finished")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
