import fasteners
import h5py
import pandas as pd


def save_dict_to_hdf5(dic, hf, path=""):
    """Recursively save dictionary to hdf5 file."""
    for key, item in dic.items():
        key_path = f"{path}/{key}" if path else key  # Create a nested path in HDF5 file

        if isinstance(item, pd.DataFrame):
            # Convert DataFrame to CSV and store as a string
            csv_string = item.to_csv(index=True)
            hf.create_dataset(key_path, data=csv_string)
        elif isinstance(item, dict):
            # Recursively save dictionary
            save_dict_to_hdf5(item, hf, key_path)
        else:
            # Directly save other data types
            hf[key_path] = item


def write_to_hdf5_with_lock(data, output_file, key, lock_file):
    lock = fasteners.InterProcessLock(lock_file)
    locked = lock.acquire(blocking=True)
    if not locked:
        raise RuntimeError("Could not acquire the lock")

    try:
        with h5py.File(output_file, "a") as f:
            save_dict_to_hdf5(data, f, key)
    finally:
        lock.release()
