# General utility functions

# Imports

import configparser
import pandas as pd
import numpy as np

from ast import literal_eval
from openbabel import openbabel
from rdkit import Chem  # Trying RDKit instead of OpenBabel


# Useful Functions

def array_in_nd_array(test, array):
    """Checks whether or not a test 1D array is contained in a full N-D array.
    Returns True if the test array is equal to any dimension of the N-D array.
    Returns False if the test array does not match a dimension of the N-D
        array.

    Inputs:
        test
        array
    Outputs:
        True/False
    """
    return any(np.array_equal(item, test) for item in array)


def canonicalize(smiles, engine="openbabel"):
    """Standardizes SMILES strings into canonical SMILES strings through
    the specified engine.
    (Important in optimizing prediction results.)

    Input:
    Output:
    """
    if engine == "openbabel":
        obconversion = openbabel.OBConversion()
        obconversion.SetInAndOutFormats("smi", "can")
        obmol = openbabel.OBMol()
        obconversion.ReadString(obmol, smiles)
        outMDL = obconversion.WriteString(obmol)
        can = outMDL.rstrip()
    elif engine == "rdkit":
        mol = Chem.MolFromSmiles(smiles)  # , sanitize=True)
        can = Chem.MolToSmiles(mol)
    else:
        raise AttributeError("Engine must be either 'openbabel' or 'rdkit'.")
    return can


def clean_inchi(df, drop_na=True, col='InChI'):
    """Cleans InChI strings in a dataframe.

    Given a dataframe with a column containing InChI information, strips
    padded/erroneous characters (e.g. newline) from the string.
    Optionally drops any rows containing NaN values.

    Args:
        df

    Returns:
        df
    """
    inchis = list()
    for inchi in df[col]:
        try:
            inchis.append(inchi.rstrip())
        except:
            inchis.append(inchi)
    df[col] = inchis
    return df


def check_elements(string):
    """Checks for chemical letters outside of the CHNOPS set.
    If the string only contains CHNOPS, returns True.
    Otherwise, returns False.
    Note: does not cover Scandium :(
    """

    bad_elements = "ABDEFGIKLMRTUVWXYZsaroudlefgibtn"  # chem alphabet -CHNOPS
    return not any(n in bad_elements for n in string)


# Config file processing functions

def read_config(file):
    """Reads config.ini file
    """
    config = configparser.ConfigParser()
    config.read(file)
    return config


def parse_config_sect(config, sect):
    """Parses a config.ini section into a dictionary
    """
    sect = config[sect]
    sectdict = {}
    for var in sect:
        try:
            sectdict[var] = literal_eval(sect[var])
        except:
            sectdict[var] = sect[var]
    return sectdict


def get_config_sects(config, remove_default=True):
    """Returns list of sections in a config file.

    Args:
        config:
        remove_default: bool, default True
            Ignores DEFAULTSECT and excludes from output dictionary.

    Returns:
    """
    sects = [sect for sect in config]
    if remove_default is True:
        sects = [sect
                 for sect in sects
                 if sect not in configparser.DEFAULTSECT]
    return sects


def config_to_dict(config, **kwargs):
    """Converts parsed config file to nested dictionary format.

    Args:
        config:
        remove_default: bool, default True
            Ignores DEFAULTSECT and excludes from output dictionary.

    Returns:
    """
    sects = get_config_sects(config, **kwargs)
    sectdict = {sect: parse_config_sect(config, sect) for sect in sects}
    return sectdict
