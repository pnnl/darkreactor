# Module for reading molecule data (InChI) and applying DarkChem rxn vectors

# Initializations

import numpy as np
import pandas as pd
import os

import darkchem
from darkreactor.darkreactor import (utils, convert)

from datetime import datetime
from openbabel import openbabel
from rdkit import Chem  # rdkit is slower than openbabel
from sklearn.model_selection import train_test_split


# Functions

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
    return utils.canonicalize(reduced, engine=engine)


def populate_products(df):
    """Takes a dataframe with SMILES strings in one column.
    Creates new column containing SMILES strings with all aromatic carbons
        converted into aliphatic carbons.
    Returns dataframe with new column appended.
    """
    df["SMILES, Product"] = [reduce_benzenoid(smiles)
                             for smiles in df["SMILES"]]
    return df


def compute_embeddings(df, col):
    """Given a dataframe and column containing canonical SMILES strings,
    converts SMILES to DarkChem character embeddings.

    Args:
    Returns:
    """
    vecs = [convert.can_to_embedding(smiles) for smiles in df[col]]
    return vecs


def populate_latent_vectors(df, model, cols):
    """Takes a dataframe containing one reactant and one product column,
        containing reactants and products in canonical SMILES format.
    Creates new columns containing latent space vector representations of
        reactant molecules and product molecules, respectively.
    Returns dataframe with new column appended.

    Args:
        model: darkchem model
        cols: list or array-like
            e.g. ['reactant_smiles', 'product_smiles']
            2-item list of column names where the first item maps to
            reactant molecule SMILES and the second item maps to
            product molecule SMILES
    """
    assert len(cols) == 2, "Error: cols argumemt must contain 2 column names"

    reactant = compute_embeddings(df, cols[0])
    product = compute_embeddings(df, cols[1])

    df["Vector"] = [convert.embedding_to_latent(vec, model)
                    for vec in reactant]
    df["Vector, Product"] = [convert.embedding_to_latent(vec, model)
                             for vec in product]

    return df


def compute_reaction_vectors(df, cols):
    """Takes a dataframe containing reactant and product latent space
    vectors and computes the difference ("reaction vector"; or
    (product - reactant)) for each reactant-product pair.
    Returns list of reaction vectors.

    Args:
        cols: list or array-like
            e.g. ["reactant_vector_col", "product_vector_col"]
    Returns:
    """
    assert len(cols) == 2, "Error: cols argumemt must contain 2 column names"

    reactants = df[cols[0]].values
    products = df[cols[1]].values

    reactions = [(products[i] - reactants[i]) for i in range(len(df))]
    return reactions


def populate_reaction_vectors(df, cols, col_name='reaction_vector'):
    """Populates dataframe with reaction vectors
    """
    df[col_name] = compute_reaction_vectors(df, cols=cols)
    return df


def apply_reaction(reactant, vec):
    """Takes a reactant and a reaction vector. Applies vector to those
        reactions.
    Returns a list of resultant vector sum.
    """
    vecsum = reactant + vec
    return vecsum


def average_vector(df, indices, col="reaction_vector"):#classes=False, col="Reaction Vectors"):
    """Computes average vector given list of indices in a dataframe
    """
    avg = np.mean(df[col].values[indices], axis=0)
    #std = np.std(df[col].values[indices], axis=0)
    return avg


# Adjust to the way that VAE literature does the reaction vector-- take mean of all mols first
def binwise_train_test(df, col="Class", combine=True,
                       random_state=None, test_size=None):
    """Splits data into specified bins and selects train/test split
    for each class.
    (To ensure bins are equally represented in train/test sets)

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
    if combine:
        train, test = np.hstack(train), np.hstack(test)  # flattens arrays
    return train, test


def self_reconstruct(mol, model, k=1, engine='openbabel'):
    """Self-reconstructs an input molecule thru DarkChem

    Args:
    Returns:
    """
    embed = convert.can_to_embedding(mol)
    encoded = convert.embedding_to_latent(embed, model)
    decoded = convert.latent_to_can(encoded, model, k=k, engine=engine)
    return decoded


# Liang's work: check for reconstructibility
def check_reconstruction(mol, arr,
                         simple=False,
                         by='smiles',
                         engine='openbabel'):
    """Checks DarkChem reconstruction (True or False) of a molecule.
    Can perform simple check (returns bool) or check vs array of
    reconstructions (returns array of bools).

    Args:
        simple: bool, default False
            Performs simple reconstruction check.
        by: str
            Molecular representation with which to compare
            Accepts: ('smiles', 'inchi', 'inchikey')
    Returns:
        bool
    """
    if simple:
        return any(arr)
    if not any(arr):
        return False
    if by != 'smiles':
        mol = convert.can_to_inchi(mol, engine=engine)
        arr = convert.cans_to_inchis(arr, engine=engine)
        if by == 'inchikey':
            mol = convert.inchi_to_key(mol, engine=engine)
            arr = convert.inchis_to_keys(arr, engine=engine)
    return (mol in arr)


def reconstruction_filter(mol, model,
                          k=1, by='smiles', engine='openbabel', simple=False):
    """Checks reconstruction of a molecule in DarkChem.
    Args:
    Returns:
    """
    reconstructed = self_reconstruct(mol, model, k=k)
    valid = check_reconstruction(mol, reconstructed,
                                 by=by, simple=simple, engine=engine)
    # if valid:
    return valid


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
    data = darkreactor.utils.clean_inchi(data)
    data = data[data["Class"].notna()].reset_index(drop=True)
    filtered = data.groupby('Class')['Class'].filter(lambda x: len(x) >= min_class_size)
    data = data[data['Class'].isin(filtered)].reset_index(drop=True)
    data["SMILES Length"] = [len(smi) for smi in data["SMILES"]]
    data = data[data["SMILES Length"] <= max_smiles_length].reset_index(drop=True)

    # Create products column
    data = populate_products(data)
    data["InChI, Product"] = [darkreactor.convert.can_to_inchi(can) for can in data["SMILES, Product"]]
    data["InChIKey, Product"] = [darkreactor.convert.inchi_to_inchikey(inchi) for inchi in data["InChI, Product"]]

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
    data["SMILES, Predicted Product"] = [darkreactor.convert.latent_to_can(vec, k=k, engine=engine) for vec in data["Vector, Predicted Product"]]
    data["InChI, Predicted Product"] = [darkreactor.convert.can_array_to_inchi_array(can, engine=engine) for can in data["SMILES, Predicted Product"]]
    data["InChIKey, Predicted Product"] = [darkreactor.convert.inchi_array_to_inchikey_array(inchis, engine=engine) for inchis in data["InChI, Predicted Product"]]

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
