"""
collect_bf_result.py

This script is used to collect the beamforming results from the output files of the beamforming simulation to adapt for MCMC algorithm.
"""
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import h5py
import pandas as pd
from loguru import logger


def collect_bf_result(bf_result_file: Path) -> defaultdict:
    res = {}
    with h5py.File(bf_result_file, "r") as f:
        all_keys = list(f.keys())
        for key_str in all_keys:
            key = eval(key_str)  # pylint: disable=eval-used
            index_list, station, position, position_index = (
                key[:-2],
                key[-2],
                tuple(list(key[-1])[:-1]),
                list(key[-1])[-1],
            )
            if station not in res:
                res[station] = {}
            if position_index not in res[station]:
                res[station][position_index] = []

            res[station][position_index].append(
                {
                    "takeoff_opt": f[key_str]["takeoff_opt"][()],
                    "azi_opt": f[key_str]["azi_opt"][()],
                    "v_opt": f[key_str]["v_opt"][()],
                    "amplitude_opt": f[key_str]["amplitude_opt"][()],
                    "azi_theoritical": f[key_str]["azi_theoritical"][()],
                    "v_theoritical": f[key_str]["v_theoritical"][()],
                    "takeoff_theoritical": f[key_str]["takeoff_theoritical"][()],
                    "index_list": index_list,
                    "position": position,
                }
            )
    return res


@click.command()
@click.option("--bf_result_dir", type=click.Path(exists=True))
@click.option("--output_file", type=click.Path())
@click.option(
    "--num_workers",
    default=5,
    help="The number of workers to collect the results",
    type=int,
)
def main(bf_result_dir: str, output_file: str, num_workers: int):
    bf_result_dir = Path(bf_result_dir)
    output_file = Path(output_file)
    bf_result_files = sorted(list(bf_result_dir.glob("*.h5")))
    logger.info(
        f"Total number of beamforming results: {len(bf_result_files)}, start to collect the results"
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(collect_bf_result, bf_result_files))
    logger.info("Finished collecting the results, star to merge the results")

    # collect the results
    bf_res = {}
    for r in results:
        for station, station_info in r.items():
            if station not in bf_res:
                bf_res[station] = {}
            for position_index, station_position_info_list in station_info.items():
                if position_index not in bf_res[station]:
                    bf_res[station][position_index] = []
                bf_res[station][position_index].extend(station_position_info_list)
    logger.info("Finished merging the results, start to save the results")

    res = {
        "beamforming_result": bf_res,
        "arrival_info": pd.read_csv(bf_result_dir / "arrival_info.csv", index_col=0),
    }

    # save to the output file in pickle format
    with open(output_file, "wb") as f:
        pickle.dump(res, f)
    logger.info(f"Finished saving the results to {output_file}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
