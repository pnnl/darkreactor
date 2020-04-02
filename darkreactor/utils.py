# Initialization

import pandas as pd
import numpy as np
import darkchem

from openbabel import openbabel
from rdkit import Chem # Trying RDKit instead of OpenBabel
#from rdkit.Chem.Draw import MolToImage


# Useful Functions

def array_in_nd_array(test, array):
    """Checks whether or not a test 1D array is contained in a full N-D array.
    Returns True if the test array is equal to any dimension of the N-D array.
    Returns False if the test array does not match a dimension of the N-D array.

    Inputs:
        test
        array
    Outputs:
        True/False
    """
    return any(np.array_equal(item, test) for item in array)


def check_elements(string):
    """Checks for chemical letters outside of the CHNOPS set.
    If the string only contains CHNOPS, returns True.
    Otherwise, returns False.
    Note: does not cover Scandium :(
    """

    bad_elements = "ABDEFGIKLMRTUVWXYZsaroudlefgibtn" # chem alphabet -CHNOPS
    return not any(n in bad_elements for n in string)


def canonicalize(smiles, engine="openbabel"):
    """Standardizes SMILES strings into canonical SMILES strings through
    OpenBabel.
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
        mol = Chem.MolFromSmiles(smiles)#, sanitize=True)
        can = Chem.MolToSmiles(mol)
    else:
        raise AttributeError("Engine must be either 'openbabel' or 'rdkit'.")
    return can
