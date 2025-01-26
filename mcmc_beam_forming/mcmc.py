"""
mcmc.py

This program performs the MCMC simulation to find the best trustworthiness of the PS arrival time.
"""
import pickle
from pathlib import Path
from typing import List

import click
import jax
import numpy as np
import pymc as pm
import pymc.math as pmath
from loguru import logger
from numba import njit
from numba.typed import List as nbList  # pylint: disable=no-name-in-module


def load_data(beamforming_pickle_file: Path, num_divide):
    with open(beamforming_pickle_file, "rb") as f:
        data = pickle.load(f)["beamforming_result"]

    all_divide_keys = []
    for station in data:
        for position_index in data[station]:
            all_divide_keys.append((station, position_index))
    all_divide_keys.sort()
    divide_keys = np.array_split(all_divide_keys, num_divide)

    return data, divide_keys


@njit
def generate_indicator_matrix(n_experiments, all_indexes):
    cache_indexes = set([0])
    cache_indexes.remove(0)
    for each in all_indexes:
        cache_indexes = cache_indexes.union(set(each))
    cache_indexes_list = sorted(cache_indexes)
    trueidx2idx = {}
    idx2trueidx = {}
    for i, each in enumerate(cache_indexes_list):
        trueidx2idx[each] = i
        idx2trueidx[i] = each

    indicator_matrix = np.zeros((n_experiments, len(cache_indexes)), dtype=np.float64)
    for i, each in enumerate(all_indexes):
        for j in each:
            indicator_matrix[i, trueidx2idx[j]] = 1 / len(each)

    return indicator_matrix, idx2trueidx


def prepare_mcmc(divide_key_this_iteration: np.ndarray, data: dict):
    observations_v, observations_takeoff = [], []
    theoritical_v, theoritical_takeoff = [], []
    all_indexes = nbList()

    for each in divide_key_this_iteration:
        station = each[0]
        position_index = int(each[1])
        for each_observation in data[station][position_index]:
            observations_v.append(each_observation["v_opt"])
            observations_takeoff.append(each_observation["takeoff_opt"])
            theoritical_v.append(each_observation["v_theoritical"])
            theoritical_takeoff.append(each_observation["takeoff_theoritical"])
            all_indexes.append(nbList(each_observation["index_list"]))

    indicator_matrix, idx2trueidx = generate_indicator_matrix(
        len(observations_v), all_indexes
    )
    return (
        observations_v,
        observations_takeoff,
        theoritical_v,
        theoritical_takeoff,
        indicator_matrix,
        idx2trueidx,
    )


def perform_mcmc(
    observations_v: List[int],
    observations_takeoff: List[int],
    theoritical_v: List[int],
    theoritical_takeoff: List[int],
    indicator_matrix: np.ndarray[float],
    sigma_good_v=1.0,
    sigma_bad_v=3.0,
    sigma_good_takeoff=15,
    sigma_bad_takeoff=45,
    warm_up_steps=1000,
    draw_steps=1000,
    chain=1,
):
    n_experiment, n_trustworthiness = indicator_matrix.shape
    with pm.Model() as model:
        # Priors
        R = pm.Beta("R", alpha=5, beta=1, shape=n_trustworthiness)
        N_good_v = pm.Normal.dist(
            mu=theoritical_v, sigma=sigma_good_v, shape=n_experiment
        )
        N_bad_v = pm.Normal.dist(
            mu=theoritical_v, sigma=sigma_bad_v, shape=n_experiment
        )
        N_good_takeoff = pm.Normal.dist(
            mu=theoritical_takeoff, sigma=sigma_good_takeoff, shape=n_experiment
        )
        N_bad_takeoff = pm.Normal.dist(
            mu=theoritical_takeoff, sigma=sigma_bad_takeoff, shape=n_experiment
        )

        # calculate the mean of trustworthiness
        R_mean = pm.math.dot(indicator_matrix, R)

        # construct the likelihood
        weight = pmath.stack([R_mean, 1 - R_mean], axis=1)
        pm.Mixture(
            "Like_v",
            w=weight,
            comp_dists=[N_good_v, N_bad_v],
            observed=observations_v,
        )
        pm.Mixture(
            "Like_takeoff",
            w=weight,
            comp_dists=[N_good_takeoff, N_bad_takeoff],
            observed=observations_takeoff,
        )
        trace = pm.sample(
            draws=draw_steps,
            tune=warm_up_steps,
            chains=chain,
            progressbar=True,
            nuts_sampler="numpyro",
        )

    return trace


@click.command()
@click.option("--beamforming_pickle_file", type=click.Path(exists=True))
@click.option("--output_directory", type=click.Path())
@click.option(
    "--num_divide",
    default=5,
    help="The number of the divide, used to reduce GPU memory requirement, default to 5.",
    type=int,
)
@click.option("--chain", default=1, help="The number of the MCMC chain, default to 1.")
@click.option("--current_divide", default=0, help="The current divide, default to 0.",type=int)
def main(
    beamforming_pickle_file: str,
    output_directory: str,
    num_divide: int,
    chain: int,
    current_divide: int,
):
    beamforming_pickle_file = Path(beamforming_pickle_file)
    output_directory = Path(output_directory)
    # make the output directory if not exist
    output_directory.mkdir(exist_ok=True)
    logger.info(f"JAX devices: {jax.devices()}")

    logger.info("Loading the data")
    data, divide_keys = load_data(beamforming_pickle_file, num_divide)

    logger.info(f"Start the MCMC simulation for {len(divide_keys)} divisions, current divide: {current_divide}")
    for i, divide_key_this_iteration in enumerate(divide_keys):
        if i != current_divide:
            # we skip other divisions to avoid a strange pymc memory issue
            continue

        logger.info(f"Prepare the MCMC for the {i}th iteration")
        (
            observations_v,
            observations_takeoff,
            theoritical_v,
            theoritical_takeoff,
            indicator_matrix,
            idx2trueidx,
        ) = prepare_mcmc(divide_key_this_iteration, data)

        n_experiment, n_trustworthiness = indicator_matrix.shape
        logger.info(
            f"Perform the MCMC for the {i}th iteration, Number of experiments: {n_experiment}, number of trustworthiness: {n_trustworthiness}"
        )
        trace = perform_mcmc(
            observations_v,
            observations_takeoff,
            theoritical_v,
            theoritical_takeoff,
            indicator_matrix,
            chain=chain,
        )

        logger.info(f"Save the MCMC result for the {i}th iteration")
        # save the trace
        trace.to_netcdf(str(output_directory / f"trace_{i}.nc"))

        # save the idx2trueidx
        with open(output_directory / f"idx2trueidx_{i}.pkl", "wb") as f:
            pickle.dump(dict(idx2trueidx), f)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
