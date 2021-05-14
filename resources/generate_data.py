"""generate_data : Generates data populated with vectors, etc.

Generates a processed data file from a raw data file, including
DarkChem latent space vectors and reaction vectors. Does not
perform any prediction tasks, but labels reconstructible molecules.
Does not delete any entries.

author: @christinehc
"""

# imports
from os.path import isfile, join

import numpy as np
import pandas as pd
import darkchem
from openbabel import openbabel
from darkreactor import (convert, react, utils)


# read config file and extract parameter dictionaries
config = utils.read_configdict('config.ini')
paths = config['paths']
params = config['parameters']
cols = config['columns']
reconstruct = config['reconstruct']

# stop openbabel/numpy error logging flood
openbabel.obErrorLog.StopLogging()  # or openbabel.obErrorLog.SetOutputLevel(0)
np.seterr(divide='ignore')

# set paths and data
file_path = paths['base']   # os.getcwd()
model_path = join(paths['model_dir'], paths['model'])
model = darkchem.utils.load_model(model_path)
try:
    data = pd.read_csv(
        join(file_path, "data", paths['data'])
        )
except UnicodeDecodeError:
    data = pd.read_pickle(
        join(file_path, "data", paths['data'])
        )

# delineate input data columns as reactants
data = data.add_suffix('_r')

# clean data, populate identifiers, filter by length
data = utils.clean_inchi(data, col=cols['inchi_r'])  # strips hidden characters

if cols['smi_r'] not in data.columns:
    data[cols['smi_r']] = [convert.inchi_to_can(inchi, engine=params['engine'])
                           for inchi in data[cols['inchi_r']]]

data["len_smiles"] = [len(smiles) for smiles in data[cols['smi_r']]]
data = data[
    data['len_smiles'] <= params['max_len_smiles']
    ].reset_index(drop=True)
data['length_bin'] = pd.cut(data['len_smiles'],
                            bins=[(10 * i) for i in range(11)])

# drop erroneous smiles
data = data[data[params['bins']].notna()].reset_index(drop=True)

# creates products (aromatic to aliphatic ring reduction) if missing
if params['perform_reaction']:
    data = react.populate_products(data,
                                   in_col=cols['smi_r'],
                                   out_col=cols['smi_p'])
else:
    assert data[cols['smi_p']], (
        f"No [{cols['smi_p']}] column detected."
        )

# remove outliers with failed reductions
data['c_count_r'] = [smi.count("c") for smi in data[cols['smi_r']]]
data['c_count_p'] = [smi.count("c") for smi in data[cols['smi_p']]]
data = data.loc[(data['c_count_r'] != 0)
                & (data['c_count_p'] == 0), :].reset_index(drop=True)

# check if molecules are reconstructible in darkchem
data['smiles_reconstruct_r'] = [
    react.self_reconstruct(
        smiles, model, k=reconstruct['k'], engine=params['engine']
        )
    for smiles in data[cols['smi_r']]
    ]

# check reactant (precursor) molecules
data['valid_reconstruct_r'] = [
    react.check_reconstruction(
        smiles, arr,
        simple=reconstruct['simple'],
        by=reconstruct['by'],
        engine=params['engine']
        )
    for smiles, arr in zip(data[cols['smi_r']], data['smiles_reconstruct_r'])
    ]

# reconstruct product molecules
data['smiles_reconstruct_p'] = [
    react.self_reconstruct(
        smiles, model, k=reconstruct['k'], engine=params['engine']
        )
    for smiles in data[cols['smi_p']]
    ]

# check product molecules
data['valid_reconstruct_p'] = [
    react.check_reconstruction(
        smiles, arr,
        simple=reconstruct['simple'],
        by=reconstruct['by'],
        engine=params['engine']
        )
    for smiles, arr in zip(data[cols['smi_p']], data['smiles_reconstruct_p'])
    ]

# create products column if missing
if cols['inchi_p'] not in data.columns:
    data[cols['inchi_p']] = [convert.can_to_inchi(can)
                             for can in data[cols['smi_p']]]
    data[cols['key_p']] = [convert.inchi_to_key(inchi)
                           for inchi in data[cols['inchi_p']]]

# compute latent space vectors and compute reaction vectors
data = react.populate_latent_vectors(data, model,
                                     in_cols=[cols['smi_r'], cols['smi_p']],
                                     out_cols=['vec_r', 'vec_p'])
data['vec_rxn'] = react.compute_reaction_vectors(data, cols=['vec_r', 'vec_p'])

# generate output files
suffix = 0
while isfile(join(file_path, "data", f"{paths['outdata']}_{suffix}.pkl")):
    suffix += 1
savename = f"{paths['outdata']}_{suffix:02d}"
data.to_pickle(join(file_path, "data", f"{savename}.pkl"))

# write config to file
with open(join(file_path, "data", f"{savename}.ini"), 'w') as cfile:
    config.write(cfile)
