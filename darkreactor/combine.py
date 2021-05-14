"""combine : Module for combined reaction vector analysis.

author: @christinehc
"""

# Imports
import numpy as np


# Functions
def weight_bin_vectors(bin_vecs, bin_name, results,
                       column='Weighted Average Vector'):
    """Compute frequency-weighted class vectors from output.

    Parameters
    ----------
    bin_vecs : pandas.DataFrame
        DataFrame loaded from classvecs.pkl.
    bin_name : str
        Name of bin used to stratify vector dataset.
    results : pandas.DataFrame
        DataFrame loaded from results.pkl.
    column : str (default: 'Weighted Average Vector')
        Name of new column containing vector outputs.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing new vector column.

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
    """Compute total weighted average reaction vector from output.

    Uses the results.pkl and classvecs.pkl outputs of a DarkReactor
    computation to generate vector.

    Parameters
    ----------
    bin_vecs : pandas.DataFrame
        DataFrame loaded from classvecs.pkl.
    results : pandas.DataFrame
        DataFrame loaded from results.pkl.
    column : str (default: 'Weighted Average Vector')
        Name of new column containing vector outputs.

    Returns
    -------
    numpy.array
        Weighted average vector.

    """
    classvecs = weight_bin_vectors(bin_vecs, results, column=column)
    avg_vec = np.sum(classvecs['Weighted Average Vector'])

    return avg_vec
