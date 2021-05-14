"""convert : Module for molecular conversions.

author: @christinehc
"""

# Imports
import numpy as np
from darkreactor import utils
from darkchem.utils import beamsearch, struct2vec, vec2struct
from openbabel import openbabel
from rdkit import Chem   # note: rdkit is slower than openbabel


# Functions
def inchi_to_can(inchi, engine="openbabel"):
    """Convert InChI to canonical SMILES.

    Parameters
    ----------
    inchi : str
        InChI string.
    engine : str (default: "openbabel")
        Molecular conversion engine ("openbabel" or "rdkit").

    Returns
    -------
    str
        Canonical SMILES.
    """
    if engine == "openbabel":
        obconversion = openbabel.OBConversion()
        obconversion.SetInAndOutFormats("inchi", "can")
        obmol = openbabel.OBMol()
        obconversion.ReadString(obmol, inchi)
        outinchi = obconversion.WriteString(obmol)
        can = outinchi.rstrip()
    elif engine == "rdkit":
        mol = Chem.MolFromInchi(inchi)
        can = Chem.MolToSmiles(mol)
    else:
        raise AttributeError(
            "Engine must be either 'openbabel' or 'rdkit'."
            )
    return can


def can_to_inchi(can, engine="openbabel"):
    """Convert canonicalized SMILES to InChI.

    Parameters
    ----------
    can : str
        Canonical SMILES.
    engine : str (default: "openbabel")
        Molecular conversion engine ("openbabel" or "rdkit").

    Returns
    -------
    str
        InChI string.
    """
    if engine == "openbabel":
        obconversion = openbabel.OBConversion()
        obconversion.SetInAndOutFormats("can", "inchi")
        obmol = openbabel.OBMol()
        obconversion.ReadString(obmol, can)
        outcan = obconversion.WriteString(obmol)
        inchi = outcan.rstrip()
    elif engine == "rdkit":
        mol = Chem.MolFromSmiles(can)
        inchi = Chem.MolToInchi(mol)
    else:
        raise AttributeError("Engine must be either 'openbabel' or 'rdkit'.")
    return inchi


def cans_to_inchis(array, **kwargs):
    """Convert array of canonical SMILES to array of InChIs.

    Parameters
    ----------
    array : list of str or array of str
        List of SMILES.
    **kwargs : dict
        Keyword arguments for `can_to_inchi`.

    Returns
    -------
    list
        List of InChI corresponding to input list of SMILES.

    """
    return [can_to_inchi(inchi, **kwargs) for inchi in array]


def inchi_to_key(inchi, engine="openbabel"):
    """Convert InChI representation to InChIKey hash.

    Parameters
    ----------
    inchi : str
        InChI representation.
    engine : str (default: "openbabel")
        Molecular conversion engine ("openbabel" or "rdkit").

    Returns
    -------
    str
        InChIKey hash.

    """

    if engine == "openbabel":
        obconversion = openbabel.OBConversion()
        obconversion.SetInAndOutFormats("inchi", "inchi")
        obmol = openbabel.OBMol()
        obconversion.ReadString(obmol, inchi)
        obconversion.SetOptions("K", obconversion.OUTOPTIONS)
        key = obconversion.WriteString(obmol).rstrip()
    elif engine == "rdkit":
        mol = Chem.MolFromInchi(inchi)
        key = Chem.MolToInchiKey(mol)
    else:
        raise AttributeError("Engine must be either 'openbabel' or 'rdkit'.")
    return key


def inchis_to_keys(array, **kwargs):
    """Convert array of InChIs to corresponding array of InChIKeys.

    Parameters
    ----------
    array : list of str or array of str
        Array of InChI strings.
    **kwargs : dict
        Keyword arguments for `inchi_to_key`.

    Returns
    -------
    list
        List of InChIKeys corresponding to input InChIs.

    """
    return [inchi_to_key(inchi, **kwargs) for inchi in array]


def can_to_embedding(smiles):
    """Convert canonical SMILES to DarkChem character embeddings.

    Parameters
    ----------
    smiles : str
        Canonical SMILES string.

    Returns
    -------
    numpy.ndarray
        Array of shape (100,).

    """
    return struct2vec(smiles).astype(int)


def embedding_to_latent(vec, model):
    """Convert DarkChem character embedding to latent space vector.

    Parameters
    ----------
    vec : list or array-like
        Vector of character embeddings.
    model : object
        Pre-trained model object with network weights.
        (e.g. `darkchem.network.VAE` object)

    Returns
    -------
    numpy.ndarray
        Latent space vector.

    """
    return model.encoder.predict(np.array([vec]))[0]


def latent_to_embedding(vec, model, k=10):
    """Convert latent space vector to DarkChem character embedding.

    Parameters
    ----------
    vec : list or array-like
        Vector of latent space coordinates.
    model : object
        Pre-trained model object with network weights.
        (e.g. `darkchem.network.VAE` object)
    k : int
        Beam width of the decoder.

    Returns
    -------
    numpy.ndarray
        Vector of character enbeddings.

    """
    softmax = model.decoder.predict(np.array([vec]))
    embedding = beamsearch(softmax, k=k).reshape(-1, 100)
    return embedding


def latent_to_can(vec, model, engine="openbabel", k=10):
    """Convert latent space vector to canonical smiles.

    Parameters
    ----------
    vec : list or array-like
        Vector of latent space coordinates.
    model : object
        Pre-trained model object with network weights.
        (e.g. `darkchem.network.VAE` object)
    engine : str (default: "openbabel")
        Molecular conversion engine ("openbabel" or "rdkit").
    k : int
        Beam width of the decoder.

    Returns
    -------
    numpy.ndarray of str
        Array of k canonical SMILES decodings.

    """
    embed = latent_to_embedding(vec, k=k, model=model)
    smiles = [vec2struct(vec) for vec in embed]
    smiles = np.array([utils.canonicalize(smi, engine=engine)
                       for smi in smiles])
    return smiles
