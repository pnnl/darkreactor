"""darkreact : Analyze a set of molecules using latent space arithmetic.

Uses DarkChem's latent space to mathematically map molecules into
latent space. Analogues to chemical transformations in latent space
are then assessed.

author: @christinehc
"""

# import required modules
from os.path import exists, join
from os import makedirs
from datetime import datetime

import numpy as np
import pandas as pd
from darkchem.utils import load_model
from openbabel import openbabel
from sklearn.model_selection import train_test_split
from darkreactor import convert, react, utils

# record script start time
start = datetime.now()

# read config file and extract parameter dictionary
config = utils.read_configdict('config.ini')
params = config['parameters']

# stop openbabel/numpy error logging flood
openbabel.obErrorLog.StopLogging()  # or openbabel.obErrorLog.SetOutputLevel(0)
np.seterr(divide='ignore')

# set paths and load data
file_path = config['paths']['base']
model_path = join(config['paths']['model_dir'], config['paths']['model'])
model = load_model(model_path)
try:
    data = pd.read_csv(
        join(file_path, "data", config['paths']['data'])
        )
except UnicodeDecodeError:
    data = pd.read_pickle(
        join(file_path, "data", config['paths']['data'])
        )

# check if molecules are reconstructible in darkchem
if config['reconstruct']['reconstruct'] is not False:
    # keep only if reactant and product are both reconstructible
    data = data[
        (data['valid_reconstruct_r']) & (data['valid_reconstruct_p'])
        ].reset_index(drop=True)

# stratify by class bin (if specified)
if params['bins'] == 'class_r':
    filtered = data.groupby(
        'class_r', as_index=False)['class_r'].filter(
            lambda x: (len(x) >= params['min_class_size'])
            )
    data = data[data['class_r'].isin(filtered.unique())].reset_index(drop=True)

# split data by index and bin type
i_train, i_test = train_test_split(data.index,
                                   random_state=params['random_state'],
                                   stratify=data[params['bins']])

# label test set molecules
data["test_set"] = data.index.isin(np.hstack(i_test))
train = data[~data['test_set']].reset_index(drop=True)

# create lookup dictionary for bin vectors
bin_vecs = {}
for bin_ in data[params['bins']].unique():
    bin_data = train[(train[params['bins']] == bin_)
                     & (~train['test_set'])]  # use only training set!
    assert len(bin_data[params['bins']].unique()) == 1, (
        "Index error: Multiple bins detected for single bin call"
        )
    bin_name = bin_data[params['bins']].unique()[0]
    bin_vec = np.mean(bin_data["vec_rxn"].values, axis=0)
    bin_vecs[bin_name] = np.array([bin_vec, "drop"])

# predict all products (the slow part)
if params['combine']:    # use only training set!
    avg_vec = np.mean(train['vec_rxn'].values, axis=0)
    data["vec_x"] = [react.apply_reaction(vec, avg_vec)
                     for vec in data["vec_r"]]

else:
    data["vec_x"] = [react.apply_reaction(vec, bin_vecs[b][0])
                     for vec, b in zip(data["vec_r"], data[params['bins']])]

# decode vectors
data["smiles_x"] = [convert.latent_to_can(vec, model, k=params['k'],
                                          engine=params['engine'])
                    for vec in data["vec_x"]]
data["inchi_x"] = [convert.cans_to_inchis(can, engine=params['engine'])
                   for can in data["smiles_x"]]
data["key_x"] = [convert.inchis_to_keys(inchis, engine=params['engine'])
                 for inchis in data["inchi_x"]]

# check whether predictions match actual products
data["valid_x"] = [any(array) for array in data["smiles_x"]]
data["match_smiles"] = [a in b
                        for a, b in zip(data[config['columns']['smi_p']],
                                        data["smiles_x"])]
data["match_inchi"] = [a in b
                       for a, b in zip(data[config['columns']['inchi_p']],
                                       data["inchi_x"])]
data["match_key"] = [a in b
                     for a, b in zip(data[config['columns']['key_p']],
                                     data["key_x"])]

# reformat and prepare data for export
bin_vecs_df = pd.DataFrame.from_dict(bin_vecs,
                                     orient="index").drop(columns=[1])
bin_vecs_df = bin_vecs_df.reset_index()
bin_vecs_df = bin_vecs_df.rename(columns={"index": params['bins'],
                                          0: "avg_vec"})

# save outputs
run = 1
while exists(f"{file_path}/results/{run:02d}"):
    run += 1
run_path = f"{file_path}/results/{run:02d}"
makedirs(run_path)

bin_vecs_df.to_pickle(
    join(run_path, f"{str(params['bins']).lower()}_vecs.pkl")
    )
data.to_pickle(join(run_path, "results.pkl"))

# compute total run time and add as field in config file
end = datetime.now()
run_time = end - start
config['postrun'] = {'run': run, 'jobtime': str(run_time)}
with open(join(run_path, "config.ini"), 'w') as configfile:    # save
    config.write(configfile)

# summarize results in log file
with open(join(run_path, "log.txt"), "w") as logfile:
    logfile.write(f"Parameters for Run {run}\n")
    logfile.write("-------------------------\n")
    logfile.write(f"Completed on: {end}\n")
    logfile.write("=====================================\n")
    logfile.write(f"Max SMILES Length:\t{params['max_len_smiles']}\n")
    logfile.write(f"Min Class Size:\t{params['min_class_size']}\n")
    logfile.write(f"Random State:\t{params['random_state']}\n")
    logfile.write(f"Combined Classes for Avg Vec?:\t{params['combine']}\n")
    logfile.write(f"Beamsearch Parameter:\t{params['k']}\n")
    logfile.write(f"Canonicalization Engine:\t{params['engine']}\n")
    logfile.write(f"Iterations:\t{params['iterations']}\n")
    if params['other_notes'] is not False:
        logfile.write(f"\nOther Notes:\t{params['other_notes']}")
    logfile.write(f"Total Job Run Time (h): {run_time}")
