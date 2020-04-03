# Molecule conversions module

# Initializations

import numpy as np
import darkchem
import darkreactor

from openbabel import openbabel
from rdkit import Chem # rdkit is slower than openbabel


# Functions

def inchi_to_can(inchi, engine="openbabel"):
    """Converts InChI to canonical SMILES.

    Args:
        inchi : str
            InChI string
        engine : "openbabel" or "rdkit", default "openbabel"
            Specify the conversion engine (OpenBabel or RDKit)
    Returns:
        smiles : canonical SMILES
    """
    if engine == "openbabel":
        obconversion = openbabel.OBConversion()
        obconversion.SetInAndOutFormats("inchi", "can")
        obmol = openbabel.OBMol()
        obconversion.ReadString(obmol, inchi)
        outinchi = obconversion.WriteString(obmol)
        can = outinchi.rstrip()
    elif engine == "rdkit":
        mol = Chem.MolFromInchi(inchi)#, sanitize=True)
        can = Chem.MolToSmiles(mol)
    else:
        raise AttributeError("Engine must be either 'openbabel' or 'rdkit'.")
    return can


def can_to_inchi(can, engine="openbabel"):
    """Converts canonicalized SMILES string into InChI representation using
    OpenBabel. (Important in verifying whether string-represented molecules are
    identical)

    Args:
        can : str
            Canonical SMILES string
        engine : "openbabel" or "rdkit", default "openbabel"
            Specify the conversion engine (OpenBabel or RDKit)
    Returns:
        inchi : str
            InChI string
    """
    if engine == "openbabel":
        obconversion = openbabel.OBConversion()
        obconversion.SetInAndOutFormats("can", "inchi")
        obmol = openbabel.OBMol()
        obconversion.ReadString(obmol, can)
        outcan = obconversion.WriteString(obmol)
        inchi = outcan.rstrip()
    elif engine == "rdkit":
        mol = Chem.MolFromSmiles(can)#, sanitize=True)
        inchi = Chem.MolToInchi(mol)
    else:
        raise AttributeError("Engine must be either 'openbabel' or 'rdkit'.")
    return inchi


def can_array_to_inchi_array(array, **kwargs):
    """Converts array of canonical SMILES to corresponding array of InChIs.

    Args:
        array : list or array
    Returns:
    """
    return [can_to_inchi(inchi, **kwargs) for inchi in array]


def inchi_to_inchikey(inchi, engine="openbabel"):
    """Converts InChI representation to InChIKey hash.

    Args:
    Returns:
    """
    if engine == "openbabel":
        obconversion = openbabel.OBConversion()
        obconversion.SetInAndOutFormats("inchi", "inchi")
        obmol = openbabel.OBMol()
        obconversion.ReadString(obmol, inchi)
        obconversion.SetOptions("K", obconversion.OUTOPTIONS)
        inchikey = obconversion.WriteString(obmol).rstrip()
    elif engine == "rdkit":
        mol = Chem.MolFromInchi(inchi)#, sanitize=True)
        inchikey = Chem.MolToInchiKey(mol)
    else:
        raise AttributeError("Engine must be either 'openbabel' or 'rdkit'.")
    return inchikey


def inchi_array_to_inchikey_array(array, **kwargs):
    """Converts array of InChIs to corresponding array of InChIKeys.

    Args:
    Returns:
    """
    return [inchi_to_inchikey(inchi, **kwargs) for inchi in array]


def smiles_to_darkchem_embedding(smiles):
    """Converts canonical SMILES strings to character embeddings using
    DarkChem.

    Args:
        smiles :
    Returns:
    """
    return darkchem.utils.struct2vec(smiles).astype(int)


def embedding_to_latent_vector(vec):
    """Converts a molecule represented as a DarkChem character embedding into
    a vector representation in DarkChem's latent space.

    Args:
    Returns:
    """
    return model.encoder.predict(np.array([vec]))[0]


def compute_embeddings(df, col):
    """Given a dataframe and column containing canonical SMILES strings,
    converts SMILES to DarkChem character embeddings.

    Args:
    Returns:
    """
    vecs = [smiles_to_darkchem_embedding(smiles) for smiles in df[col]]
    return vecs


def latent_to_embedding(vec, k=10):
    """Converts latent space vector to character embedding.
    Args:
        k : int
            beamsearch parameter
    Returns:
    """
    softmax = model.decoder.predict(np.array([vec]))
    embed = darkchem.utils.beamsearch(softmax, k=k).reshape(-1,100)
    return embed


def latent_to_can(vec, engine="openbabel", k=10):
    """Converts latent space vector to canonical smiles.
    Args:

    Returns:
    """
    embed = latent_to_embedding(vec, k=k)
    smiles = [darkchem.utils.vec2struct(vec) for vec in embed]
    smiles = np.array([darkreactor.utils.canonicalize(smi, engine=engine) for smi in smiles])
    return smiles
