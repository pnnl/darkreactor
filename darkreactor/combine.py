"""
darkreactor.combine
-------------------
Module for combined reaction vector analysis
"""

# Imports
import numpy as np
import pandas as pd


# Functions
def weight_class_vectors(classvecs, results,
                         column='Weighted Average Vector'):
    """Given the results.pkl and classvecs.pkl outputs of a DarkReactor
    computation, computes class vectors re-weighted by frequency.

    Args:
        classvecs: pandas.DataFrame
            DataFrame loaded from classvecs.pkl
        results: pandas.DataFrame
            DataFrame loaded from results.pkl
        column: str
            Name of column to append

    Returns:
        pandas.DataFrame
            with new column appended
    """
    train = results[~results["Test Set"]]

    weight_vecs = []
    for class_name, class_vec in zip(classvecs['Class'],
                                     classvecs['Average Vector']):
        class_size = len(train[train['Class'] == class_name])
        class_weight = class_size / len(train)
        weight_vec = class_weight * class_vec
        weight_vecs.append(weight_vec)

    classvecs[column] = weight_vecs

    return classvecs


def weighted_average_vector(classvecs, results,
                            column='Weighted Average Vector'):
    """Given the results.pkl and classvecs.pkl outputs of a DarkReactor
    computation, computes a total weighted average reaction vector.

    Args:
        classvecs: pandas.DataFrame
        results: pandas.DataFrame
        column: str

    Returns:
        numpy.array
        Weighted average vector
    """
    classvecs = weight_class_vectors(classvecs, results, column=column)
    avg_vec = np.mean(classvecs['Weighted Average Vector'])

    return avg_vec
