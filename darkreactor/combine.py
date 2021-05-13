"""
darkreactor.combine
-------------------
Module for combined reaction vector analysis
"""

# Imports
import numpy as np
import pandas as pd


# Functions
def weight_bin_vectors(bin_vecs, bin_name, results,
                       column='Weighted Average Vector'):
    """Given the results.pkl and classvecs.pkl outputs of a DarkReactor
    computation, computes class vectors re-weighted by frequency.

    Args:
        bin_vecs: pandas.DataFrame
            DataFrame loaded from classvecs.pkl
        bin_name: str
            Name of the bin used to compute separate bin-wise vectors
        results: pandas.DataFrame
            DataFrame loaded from results.pkl
        column: str
            Name of column to append

    Returns:
        pandas.DataFrame
            with new column appended
    """
    train = results[~results["Test Set"]]

    weighted_vecs = []
    for a_name, a_vec in zip(bin_vecs[bin_name],
                             bin_vecs['Average Vector']):
        size = len(train[train[bin_name] == a_name])
        weight = size / len(train)
        weighted_vec = weight * a_vec
        weighted_vecs.append(weighted_vec)

    bin_vecs[column] = weighted_vecs

    return bin_vecs


def weighted_average_vector(bin_vecs, results,
                            column='Weighted Average Vector'):
    """Given the results.pkl and classvecs.pkl outputs of a DarkReactor
    computation, computes a total weighted average reaction vector.

    Args:
        bin_vecs: pandas.DataFrame
        results: pandas.DataFrame
        column: str

    Returns:
        numpy.array
        Weighted average vector
    """
    classvecs = weight_class_vectors(classvecs, results, column=column)
    avg_vec = np.sum(classvecs['Weighted Average Vector'])

    return avg_vec
