# General utility functions

# Imports
import configparser
from ast import literal_eval

import numpy as np
from openbabel import openbabel
from rdkit import Chem  # Trying RDKit instead of OpenBabel


# Useful Functions

def array_in_nd_array(test, array):
    """Check whether or not a test 1D array is contained in a full N-D array.
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
    """Standardize SMILES strings into canonical SMILES strings through
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
    """Clean InChI strings in a dataframe.

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
    """Check for chemical letters outside of the CHNOPS set.
    If the string only contains CHNOPS, returns True.
    Otherwise, returns False.
    Note: does not cover Scandium :(
    """

    bad_elements = "ABDEFGIKLMRTUVWXYZsaroudlefgibtn"  # chem alphabet -CHNOPS
    return not any(n in bad_elements for n in string)


# Config file processing functions

def read_config(filename):
    """Read config.ini file.

    Creates configparser.ConfigParser() object.

    Parameters
    ----------
    filename : .ini
        /path/to/config.ini

    Returns
    -------
    configparser.ConfigParser()
        ConfigParser() object containing config data

    """
    config = configparser.ConfigParser()
    config.read(filename)
    return config


def parse_config_sect(config, sect):
    """Parse a config.ini section into a dictionary.

    Parameters
    ----------
    config : configparser.ConfigParser()
        ConfigParser() object
    sect : str
        Name of section in config file

    Returns
    -------
    dict

    """
    sect = config[sect]
    sectdict = {}
    for var in sect:
        try:
            sectdict[var] = literal_eval(sect[var])
        except (ValueError, TypeError, SyntaxError):
            sectdict[var] = sect[var]
    return sectdict


def get_config_sects(config, remove_default=True):
    """Find list of all sections in a config file.

    Parameters
    ----------
    config :
    remove_default : bool, default True
        Ignores DEFAULTSECT and excludes from output dictionary.

    Returns
    -------
    list
        List of sections contained in a config file.

    """
    sects = [sect for sect in config]
    if remove_default is True:
        sects = [sect
                 for sect in sects
                 if sect not in configparser.DEFAULTSECT]
    return sects


def config_to_dict(config, **kwargs):
    """Convert parsed config file to nested dictionary format.

    Parameters
    ----------
    config :
    remove_default : bool, default True
        Ignores DEFAULTSECT and excludes from output dictionary.

    Returns
    -------
    dict
        Dictionary-ized version of a config file, where each
        section in the config.ini is a dictionary key, and each
        parameter is a dictionary value.

    """
    sects = get_config_sects(config, **kwargs)
    sectdict = {sect: parse_config_sect(config, sect) for sect in sects}
    return sectdict


def read_configdict(filename):
    """Short summary.

    Parameters
    ----------
    filename : type
        Description of parameter `filename`.

    Returns
    -------
    type
        Description of returned object.

    """
    config = read_config(filename)
    return config_to_dict(config)
