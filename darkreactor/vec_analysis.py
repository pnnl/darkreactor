# Initializations

import numpy as np
import pandas as pd
import os

import darkchem
import darknight

from datetime import datetime
from openbabel import openbabel
from rdkit import Chem # rdkit is slower than openbabel
from sklearn.model_selection import train_test_split


# Functions

def clean_inchi(df, drop_na=True):
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
    for inchi in df["InChI"]:
        try:
            inchis.append(inchi.rstrip())
        except:
            inchis.append(inchi)
    df["InChI"] = inchis
    return df


def inchi_to_can(inchi, engine="openbabel"):
    """Converts InChI to canonical SMILES.

    The conversion engine (OpenBabel or RDKit) can be specified.

    Args:
        inchi : str
            InChI string
        engine : "openbabel" or "rdkit", default "openbabel"
            select conversion engine (OpenBabel or RDKit)
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
            canonical SMILES string
        engine : "openbabel" or "rdkit", default "openbabel"
            select conversion engine (OpenBabel or RDKit)
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
    """
    Args:
    Returns:
    """
    return [can_to_inchi(inchi, **kwargs) for inchi in array]


def inchi_to_inchikey(inchi, engine="openbabel"):
    """
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
    """
    Args:
    Returns:
    """
    return [inchi_to_inchikey(inchi, **kwargs) for inchi in array]


def reduce_benzenoid(smiles, engine="openbabel"):
    """Reduces all aromatic carbons represented in a canonical SMILES string
        into aliphatic carbon representations.

    Ensures that output molecule is a canonical SMILES

    Args:
        smiles :
        engine :
    Returns:
    """
    reduced = smiles.replace("c", "C")
    return darknight.canonicalize(reduced, engine=engine)


def populate_products(df):
    """Takes a dataframe with SMILES strings in one column.
    Creates new column containing SMILES strings with all aromatic carbons
        converted into aliphatic carbons.
    Returns dataframe with new column appended.
    """
    df["SMILES, Product"] = [reduce_benzenoid(smiles) for smiles in df["SMILES"]]
    return df


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
    """
    return model.encoder.predict(np.array([vec]))[0]


def compute_embeddings(df, col):
    """Given a dataframe and column containing canonical SMILES strings,
    converts SMILES to DarkChem character embeddings.
    """
    vecs = [smiles_to_darkchem_embedding(smiles) for smiles in df[col]]
    return vecs


def populate_latent_vectors(df, cols=["SMILES", "SMILES, Product"]):
    """Takes a dataframe containing one reactant and one product column,
        containing reactants and products in canonical SMILES format.
    Creates new columns containing latent space vector representations of
        reactant molecules and product molecules, respectively.
    Returns dataframe with new column appended.
    """
    reactant = compute_embeddings(df, cols[0])
    product = compute_embeddings(df, cols[1])

    df["Vector"] = [embedding_to_latent_vector(vec) for vec in reactant]
    df["Vector, Product"] = [embedding_to_latent_vector(vec) for vec in product]

    return df


def compute_reaction_vectors(df, cols=["Vector", "Vector, Product"]):
    """Takes a dataframe containing reactant and product latent space vectors
        and computes the difference ("reaction vector"; or (product - reactant))
        for each reactant-product pair.
    Returns list of reaction vectors.
    """
    num = len(df)
    reactants = df[cols[0]].values
    products = df[cols[1]].values

    reactions = [(products[i] - reactants[i]) for i in range(num)]
    return reactions


def populate_reaction_vectors(df, cols=["Vector", "Vector, Product"]):
    """Populates dataframe with reaction vectors
    """
    df["Reaction Vector"] = compute_reaction_vectors(df, cols=cols)
    return df


def apply_reaction(reactant, vec):
    """Takes a reactant and a reaction vector. Applies vector to those
        reactions.
    Returns a list of resultant vector sum.
    """
    vecsum = reactant + vec
    return vecsum


def average_vector(df, indices, col="Reaction Vector"):#classes=False, col="Reaction Vectors"):
    """Computes average vector given list of indices in a dataframe
    """
    avg = np.mean(df[col].values[indices], axis=0)
    #std = np.std(df[col].values[indices], axis=0)
    return avg


# Adjust to the way that VAE literature does the reaction vector-- take mean of all mols first
def classwise_train_test(df, col="Class", combine=True, random_state=None, test_size=None):
    """Splits data into classes and selects train/test split for each class.
    (To ensure equal class representation in train/test sets)

    Args:
        df :
        col :
        combine : bool
            If True, returns flattened array. If False, preserves
            array of index arrays for each class
    """
    train, test = [], []
    classes = df[col].unique()
    for aclass in classes:
        indices = df[df[col] == aclass].index.values
        i_train, i_test = train_test_split(indices, random_state=random_state,
                                                    test_size=test_size)
        train.append(i_train)
        test.append(i_test)
    train, test = np.array(train), np.array(test)
    if combine == True:
        train, test = np.hstack(train), np.hstack(test) # flattens arrays
    return train, test


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
    smiles = np.array([darknight.canonicalize(smi, engine=engine) for smi in smiles])
    return smiles


# Script to predict results using darkchem

if __name__ == "__main__":

    # Get start time
    start = datetime.now()

    # Set file directory
    filepath = os.getcwd()
    sean = "/people/colb804/deepscience/result/publication_networks"
    #filepath = "/Users/chan898/Documents/Papers/2020-DIRECT_DarkChem"

    # Set parameters
    max_smiles_length = 100
    min_class_size = 10
    random_state = 9
    combine = False
    k = 1
    iterations = 1
    engine = "openbabel"
    other_notes = False

    # Disable all but critical messages from openbabel
    openbabel.obErrorLog.SetOutputLevel(0)

    # Disable numpy "RuntimeWarning: divide by zero"
    np.seterr(divide='ignore')

    # Load model
    model = darkchem.utils.load_model(f"{sean}/N7b_[M+H]/")

    # Load DarkChem training data -not necessary
    #x = np.load(f"{filepath}/darkchem_files/combined_[M+H]_smiles.npy")
    #y = np.load(f"{filepath}/darkchem_files/combined_[M+H]_labels.npy") # must have the same number of columns as the data the network was trained on

    # Read file
    data = pd.read_csv(f"{filepath}/data/combined_[M+H]_darkchem_benzenoids.csv")

    # Clean data, remove classes with <10 molecules, filter by SMILES str length
    data = clean_inchi(data)
    data = data[data["Class"].notna()].reset_index(drop=True)
    filtered = data.groupby('Class')['Class'].filter(lambda x: len(x) >= min_class_size)
    data = data[data['Class'].isin(filtered)].reset_index(drop=True)
    data["SMILES Length"] = [len(smi) for smi in data["SMILES"]]
    data = data[data["SMILES Length"] <= max_smiles_length].reset_index(drop=True)

    # Create products column
    data = populate_products(data)
    data["InChI, Product"] = [can_to_inchi(can) for can in data["SMILES, Product"]]
    data["InChIKey, Product"] = [inchi_to_inchikey(inchi) for inchi in data["InChI, Product"]]

    # Compute latent space vectors and compute reaction vecs
    data = populate_latent_vectors(data)
    data = populate_reaction_vectors(data)

    i_train, i_test = classwise_train_test(data, random_state=random_state, combine=combine)

    data["Test Set"] = data.index.isin(np.hstack(i_test))

    # Create lookup dictionary for class vectors
    classvecs = {}
    for i_class in i_train:
        #avgvec = average_vector(data, i_class)
        classdata = data.loc[i_class]
        assert len(classdata["Class"].unique()) == 1, "Index error: Multiple classes detected for single class call"
        classname = classdata["Class"].unique()[0]
        classvec = np.mean(classdata["Reaction Vector"].values, axis=0)
        classvecs[f"{classname}"] = np.array([classvec, "drop"])

    # Predict all (the slow part)
    data["Vector, Predicted Product"] = [apply_reaction(vec, classvecs[c][0]) for vec, c in zip(data["Vector"], data["Class"])]
    data["SMILES, Predicted Product"] = [latent_to_can(vec, k=k, engine=engine) for vec in data["Vector, Predicted Product"]]
    data["InChI, Predicted Product"] = [can_array_to_inchi_array(can, engine=engine) for can in data["SMILES, Predicted Product"]]
    data["InChIKey, Predicted Product"] = [inchi_array_to_inchikey_array(inchis, engine=engine) for inchis in data["InChI, Predicted Product"]]

    # Success?!?
    data["Valid Prediction"] = [any(array) for array in data["SMILES, Predicted Product"]]
    data["SMILES Match"] = [a in b for a, b in zip(df["SMILES, Product"], data["SMILES, Predicted Product"])]
    data["InChI Match"] = [a in b for a, b in zip(df["InChI, Product"], data["InChI, Predicted Product"])]
    data["InChIKey Match"] = [a in b for a, b in zip(df["InChIKey, Product"], data["InChIKey, Predicted Product"])]

    # Prepare for export
    classvecs_df = pd.DataFrame.from_dict(classvecs, orient="index").drop(columns=[1]).reset_index().rename(columns={"index": "Class", 0:"Average Vector"})

    # Outputs
    run = 1
    while os.path.exists(f"{filepath}/results/{run}"):
        run += 1
    os.makedirs(f"{filepath}/results/{run}")
    runpath = f"{filepath}/results/{run}"

    classvecs_df.to_pickle(f"{runpath}/classvecs.pkl")
    data.to_pickle(f"{runpath}/results.pkl")
    #data.to_csv(f"{runpath}/results.csv", index=False)

    end = datetime.now()

    with open(f"{runpath}/log.txt", "w") as logfile:
        logfile.write(f"Parameters List\n")
        logfile.write("----------------------\n")
        logfile.write(f"Run #:\t{run}\n")
        logfile.write("======================\n")
        logfile.write(f"Maximum SMILES Length:\t{max_smiles_length}\n")
        logfile.write(f"Minimum Class Size:\t{min_class_size}\n")
        logfile.write(f"Random State:\t{random_state}\n")
        logfile.write(f"Combined Classes for Avg Vec?:\t{combine}\n")
        logfile.write(f"Beamsearch Parameter:\t{k}\n")
        logfile.write(f"Canonicalization Engine:\t{engine}\n")
        logfile.write(f"Iterations:\t{iterations}\n")
        if other_notes != False:
            logfile.write(f"\nOther Notes:\t{other_notes}")
        logfile.write(f"Total Job Run Time (h): {(end - start)}")
