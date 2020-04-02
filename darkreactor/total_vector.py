# Initialization

import os
import configparser

import numpy as np
import pandas as pd

import darkchem
import darknight
import darkchem_vec_analysis

from ast import literal_eval
from datetime import datetime
from openbabel import openbabel
from rdkit import Chem



# Functions

def read_config(file):
    """Reads config.ini file
    """
    config = configparser.ConfigParser()
    config.read(file)
    #sects = [sect for sect in config if sect not in configparser.DEFAULTSECT]
    return config


def parse_config_sect(config, sect):
    """Parses config.ini sections
    """
    sect = config[sect]
    sect = {var: literal_eval(sect[var]) for var in sect}
    return sect


def get_config_sects(config, remove_default=True):
    """Returns list of sections in a config file.

    Args:
        config:
        remove_default: bool, default True
            Ignores DEFAULTSECT and excludes from output dictionary.

    Returns:
    """
    sects = [sect for sect in config]
    if remove_default == True:
        sects = [sect for sect in sects if sect not in configparser.DEFAULTSECT]
    return sects


def config_to_dict(config, **kwargs):
    """Converts parsed config file to nested dictionary format.

    Args:
        config:
        remove_default: bool, default True
            Ignores DEFAULTSECT and excludes from output dictionary.

    Returns:
    """
    sects = get_config_sects(config, remove_default=remove_default)
    return sects



# Re-analyze prior analysis for single average reaction vector
if __name__ == '__main__':

    # Get start time
    start = datetime.now()

    # Set file directory
    filepath = os.getcwd()

    # Config file
    config = read_config('config.ini')

    # Get parameters
    paths = config['paths']
    paths = {path: literal_eval(paths[path]) for path in paths}
    model_path = f'{paths['model_dir']}/{paths['model']}'
    params = config['parameters']
    params = {param: literal_eval(params[param]) for param in params}
    result = 9
    resultpath = f'{filepath}/results/{result}'
    combine = True

    # Disable most openbabel and numpy errors
    openbabel.obErrorLog.SetOutputLevel(0)
    np.seterr(divide='ignore')

    # Load model
    model = darkchem.utils.load_model(f"{sean}/N7b_[M+H]/")

    # Read file
    data = pd.read_pickle(f'{resultpath}/results.pkl')
    vecs = pd.read_pickle(f'{resultpath}/classvecs.pkl')

    # Recapture training and testing sets
    i_train = data[~data["Test Set"]].index
    i_test = data[data["Test Set"]].index

    avg_vec = average_vector(data, i_train)

    # Predict all
    data["Vector, Predicted Product, Total"] = [apply_reaction(vec, avg_vec) for vec in data["Vector"]]
    data["SMILES, Predicted Product, Total"] = [latent_to_can(vec, k=k, engine=engine) for vec in data["Vector, Predicted Product, Total"]]
    data["InChI, Predicted Product, Total"] = [can_array_to_inchi_array(can, engine=engine) for can in data["SMILES, Predicted Product, Total"]]
    data["InChIKey, Predicted Product, Total"] = [inchi_array_to_inchikey_array(inchis, engine=engine) for inchis in data["InChI, Predicted Product, Total"]]

    # Success?!?
    data["Valid Prediction, Total"] = [any(array) for array in data["SMILES, Predicted Product, Total"]]
    data["SMILES Match, Total"] = [a in b for a, b in zip(data["SMILES, Product"], data["SMILES, Predicted Product, Total"])]
    data["InChI Match, Total"] = [a in b for a, b in zip(data["InChI, Product"], data["InChI, Predicted Product, Total"])]
    data["InChIKey Match, Total"] = [a in b for a, b in zip(data["InChIKey, Product"], data["InChIKey, Predicted Product, Total"])]


    total_row = {f'{vecs.columns[0]}': 'Total', f'{vecs.columns[1]}': avg_vec}
    vecs = vecs.append(total_row, ignore_index=True)
    vecs.to_pickle(f'{resultpath}/classvecs_total.pkl')

    end = datetime.now()
