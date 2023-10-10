
"""
@File    :   data_parser.py
@Date    :   2023/10/10
@Author  :   Eytan Adler
@Description : Read hydrogen property data from the data files.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================


def get_sat_property(name):
    """Get saturated hydrogen property from data file for a range of temperatures.

    Parameters
    ----------
    name : str
        Column name desired from data file

    Returns
    -------
    numpy array
        Numpy array with data from requested column
    """
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saturated_properties.txt")

    with open(file, "r") as f:
        for line in f:
            columns = line.split("\t")
            break

    if name not in columns and name + "\n" not in columns:
        raise ValueError(f"{name} is an invalid column name")
    try:
        idx = columns.index(name)
    except ValueError:
        idx = columns.index(name + "\n")

    data = np.genfromtxt(file, delimiter="\t", skip_header=1)

    return data[:, idx]

def get_property(name, phase="both", pressure=None):
    """Get property of hydrogen from data files for a range of temperatures and pressures.

    Parameters
    ----------
    name : str
        Column name desired from data file
    phase : str
        Select whether to return liquid data, vapor data, or both by setting this to
        "liquid", "vapor", or "both", by default both
    pressure : str, optional
        Desired pressure in bar formatted as a string to match the data file name, by default None
        where it will return data from all files

    Returns
    -------
    numpy array
        Flattened 1D array with data from all data files or just the specified one if pressure argument defined
    """
    # Check phase input
    if phase not in ["liquid", "vapor", "both"]:
        raise ValueError(f"Phase input must be either \"liquid\", \"vapor\", or \"both\", not \"{phase}\"")

    # Get the data file names
    data_dir = os.path.dirname(os.path.abspath(__file__))
    dir_files = os.listdir(data_dir)
    if pressure is None:
        data_files = []
        for file in dir_files:
            if "_bar_properties.txt" in file:
                data_files.append(file)
        data_files.sort()
    else:
        file = pressure + "_bar_properties.txt"
        if file not in dir_files:
            raise ValueError(f"No data file with a pressure of {pressure} found")
        data_files = [file]

    data = np.array([], dtype=float)

    for filename in data_files:
        file = os.path.join(data_dir, filename)
        with open(file, "r") as f:
            idx_first_vapor = 0
            for i, line in enumerate(f):
                if i == 0:
                    columns = line.split("\t")
                    if phase == "both":
                        break
                    continue
                # If phase isn't both, find the cutoff point
                is_liquid = "liquid" in line.split("\t")[-1]
                if is_liquid:
                    idx_first_vapor = i  # i starts at zero in the header here, so subtract an extra one
                else:
                    break

        if name not in columns and name + "\n" not in columns:
            raise ValueError(f"{name} is an invalid column name")
        try:
            idx = columns.index(name)
        except ValueError:
            idx = columns.index(name + "\n")

        # Get the data from the file
        data_cur_file = np.genfromtxt(file, delimiter="\t", skip_header=1)

        # Figure out which rows must be taken depending on the requested phase
        row_start = 0
        row_end = data_cur_file.shape[0]
        if phase == "liquid":
            row_end = idx_first_vapor
        elif phase == "vapor":
            row_start = idx_first_vapor

        # Append the data from the current file to the data
        data = np.hstack((data, data_cur_file[row_start:row_end, idx]))

    return data
