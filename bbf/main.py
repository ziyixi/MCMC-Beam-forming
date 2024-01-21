import click
import numpy as np
import pandas as pd
from loguru import logger
from mpi4py import MPI

from bbf.beamforming.single import bf_single
from bbf.bootstrap.get_grids import get_grids
from bbf.bootstrap.perform_bootstrap import analyze_bootstrap_results, bootstrap
from bbf.setting import MPI_SHUFFLE_SEED
from bbf.utils.locker_write import write_to_hdf5_with_lock


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
    "--mpi_configurations",
    default="2,4",
    help="The mpi configurations, in the format of 'number_of_nodes,number_of_cores_per_node'",
)
@click.option(
    "--output_file",
    default="bbf.h5",
    help="The output hdf5 file name, containing the results of the beamforming and bootstrap",
)
@click.option(
    "--output_lock_file",
    default="bbf.lock",
    help="The output lock file name, used to lock the hdf5 file",
)
@click.option(
    "--log_file",
    default="bbf.log",
    help="The log file name",
)
def main(
    coordinates: str,
    search_step_length: str,
    box_size: str,
    arrival_info_file: str,
    minumum_number_of_ps_events_in_box: int,
    mpi_configurations: str,
    output_file: str,
    output_lock_file: str,
    log_file: str,
):
    # * init logger
    logger.add(log_file, format="{time} {level} {message}", level="INFO")

    # * init mpi
    global_comm = MPI.COMM_WORLD
    global_rank = global_comm.Get_rank()
    global_size = global_comm.Get_size()

    # * parse the parameters
    lon0, lon1, lat0, lat1 = [float(x) for x in coordinates.split(",")]
    lon_step, lat_step, depth_step = [float(x) for x in search_step_length.split(",")]
    horizontal_size, vertical_size = [float(x) for x in box_size.split(",")]
    arrival_info = pd.read_csv(arrival_info_file)
    number_of_nodes, number_of_cores_per_node = [
        int(x) for x in mpi_configurations.split(",")
    ]
    assert number_of_nodes * number_of_cores_per_node == global_size

    # * split MPI comm
    color = global_rank // number_of_cores_per_node
    key = global_rank % number_of_cores_per_node
    local_comm = global_comm.Split(color, key)
    local_rank = local_comm.Get_rank()
    local_size = local_comm.Get_size()

    # * get the grids
    if local_rank == 0:
        grids_total = get_grids(
            lon0,
            lon1,
            lat0,
            lat1,
            lon_step,
            lat_step,
            depth_step,
            horizontal_size,
            vertical_size,
            arrival_info,
            minumum_number_of_ps_events_in_box,
            global_rank,
        )
        to_split = []
        for sta, grids_sta in grids_total.items():
            for position in grids_sta:
                to_split.append([sta, position])
        to_split = np.array(to_split, dtype=object)
        # evenly distribute the grids based on len(grids[sta][position])
        # seed the random number generator
        np.random.seed(MPI_SHUFFLE_SEED)
        np.random.shuffle(to_split)
        to_split_this_node = np.array_split(to_split, number_of_nodes)[color]
        # get grids for this node
        grids = []
        for sta, position in to_split_this_node:
            grids.append([sta, position, grids_total[sta][position]])
    else:
        grids = None
    # broadcast the grids to the same node
    grids = local_comm.bcast(grids, root=0)
    # sort grids based on sta, position
    grids = sorted(grids, key=lambda x: (x[0], x[1]))

    # * for each grid, do the bootstrap
    count = 0
    for each in grids:
        sta, position, grid_df = each
        station_lld = [grid_df["SLON"].iloc[0], grid_df["SLAT"].iloc[0]]
        # get the P wave beamforming results
        res_p = bf_single(
            grid_df,
            station_lld=station_lld,
            phase_key="P",
            comm=local_comm,
        )
        # get the PS wave beamforming results
        res_ps = bf_single(
            grid_df,
            station_lld=station_lld,
            phase_key="PS",
            comm=local_comm,
        )

        # do bootstrapping
        bootstrap_result = bootstrap(
            grid_df,
            station_lld=station_lld,
            fix_theta_in_search_value=res_ps["theoritical_azi"],
            theoritical_vp=res_ps["theoritical_vp"],
            comm=local_comm,
        )

        # analyze the bootstrap results
        misfit_threshold_list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        result_for_thresholds = analyze_bootstrap_results(
            grid_df,
            bootstrap_result,
            misfit_threshold_list,
            station_lld=station_lld,
            comm=local_comm,
        )
        # add the case for 1.0
        result_for_thresholds.append(res_ps)

        # write the results to hdf5
        output_dict = {
            "sta": sta,
            "position": position,
            "grid_df": grid_df,
            "res_p": res_p,
            "res_ps": res_ps,
            "bootstrap_result": bootstrap_result,
            "result_for_thresholds": {
                str(k): v
                for k, v in zip(misfit_threshold_list + [1.0], result_for_thresholds)
            },
        }

        count += 1
        if local_rank == 0:
            write_to_hdf5_with_lock(
                data=output_dict,
                output_file=output_file,
                lock_file=output_lock_file,
                key=f"{sta}_{'_'.join([str(x) for x in position])}",
            )
            # * logging part
            logger.info(
                f"[Node {color}] || [Grid {count}/{len(grids)}] || [Station: {sta}] || [Position: {position}] || [Events: {len(grid_df)}]"
            )

    global_comm.barrier()
    MPI.Finalize()
