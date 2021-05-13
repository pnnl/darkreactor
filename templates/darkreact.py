# Initialization

import os

import numpy as np
import pandas as pd

import darkchem
from darkreactor import darkreactor

from openbabel import openbabel
from datetime import datetime


# Script to predict results using darkchem

if __name__ == "__main__":

    # Get start time
    start = datetime.now()

    # Set file directory
    filepath = os.getcwd()
    #sean = "/people/colb804/deepscience/result/publication_networks"
    #filepath = "/Users/chan898/Documents/Papers/2020-DIRECT_DarkChem"

    # Set parameters
    config = darkreactor.utils.read_config('config.ini')
    configdict = darkreactor.utils.config_to_dict(config)
    paths = configdict['paths']
    params = configdict['parameters']
    reconstruct = configdict['reconstruct']

    # Disable all but critical messages from openbabel
    #openbabel.obErrorLog.SetOutputLevel(0)

    # Stop all openbabel logging
    openbabel.obErrorLog.StopLogging()

    # Disable numpy "RuntimeWarning: divide by zero"
    np.seterr(divide='ignore')

    # Load model
    model = darkchem.utils.load_model(f"{paths['model_dir']}/{paths['model']}")

    # Load DarkChem training data -not necessary
    #x = np.load(f"{filepath}/darkchem_files/combined_[M+H]_smiles.npy")
    #y = np.load(f"{filepath}/darkchem_files/combined_[M+H]_labels.npy") # must have the same number of columns as the data the network was trained on

    # Read file
    file = paths['data']
    data = pd.read_csv(f"{filepath}/data/{file}")

    # Clean data, remove classes with <10 molecules, filter by SMILES str length
    data = darkreactor.utils.clean_inchi(data)
    data = data[data["Class"].notna()].reset_index(drop=True)
    if 'SMILES' not in data.columns:
        data['SMILES'] = [darkreactor.convert.inchi_to_can(inchi, engine=params['engine']) for inchi in data['InChI']]
    data["SMILES Length"] = [len(smi) for smi in data["SMILES"]]
    data = data[data["SMILES Length"] <= params['max_smiles_length']].reset_index(drop=True)
    data = darkreactor.react.populate_products(data)
    if reconstruct['reconstruct'] != False:
        data['SMILES, Reconstructed'] = [darkreactor.react.self_reconstruct(smiles, k=reconstruct['k'], model=model, engine=params['engine']) for smiles in data['SMILES']]
        data['Valid Reconstruction'] = [darkreactor.react.check_reconstruction(smiles, arr, simple=reconstruct['simple'], by=reconstruct['by'], engine=params['engine']) for smiles, arr in zip(data['SMILES'], data['SMILES, Reconstructed'])]
        data['SMILES, Reconstructed, Product'] = [darkreactor.react.self_reconstruct(smiles, k=reconstruct['k'], model=model, engine=params['engine']) for smiles in data['SMILES, Product']]
        data['Valid Reconstruction, Product'] = [darkreactor.react.check_reconstruction(smiles, arr, simple=reconstruct['simple'], by=reconstruct['by'], engine=params['engine']) for smiles, arr in zip(data['SMILES, Product'], data['SMILES, Reconstructed, Product'])]
        data = data[(data['Valid Reconstruction']) & (data['Valid Reconstruction, Product'])].reset_index(drop=True)
    filtered = data.groupby('Class')['Class'].filter(lambda x: len(x) >= params['min_class_size'])
    data = data[data['Class'].isin(filtered.unique())].reset_index(drop=True)

    # Create products column
    #data = darkreactor.react.populate_products(data)
    data["InChI, Product"] = [darkreactor.convert.can_to_inchi(can) for can in data["SMILES, Product"]]
    data["InChIKey, Product"] = [darkreactor.convert.inchi_to_inchikey(inchi) for inchi in data["InChI, Product"]]

    # Compute latent space vectors and compute reaction vecs
    data = darkreactor.react.populate_latent_vectors(data, model=model)
    data = darkreactor.react.populate_reaction_vectors(data)

    i_train, i_test = darkreactor.react.classwise_train_test(data, random_state=params['random_state'], combine=params['combine'])

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
    data["Vector, Predicted Product"] = [darkreactor.react.apply_reaction(vec, classvecs[c][0]) for vec, c in zip(data["Vector"], data["Class"])]
    data["SMILES, Predicted Product"] = [darkreactor.convert.latent_to_can(vec, k=params['k'], engine=params['engine'], model=model) for vec in data["Vector, Predicted Product"]]
    data["InChI, Predicted Product"] = [darkreactor.convert.can_array_to_inchi_array(can, engine=params['engine']) for can in data["SMILES, Predicted Product"]]
    data["InChIKey, Predicted Product"] = [darkreactor.convert.inchi_array_to_inchikey_array(inchis, engine=params['engine']) for inchis in data["InChI, Predicted Product"]]

    # Success?!?
    data["Valid Prediction"] = [any(array) for array in data["SMILES, Predicted Product"]]
    data["SMILES Match"] = [a in b for a, b in zip(data["SMILES, Product"], data["SMILES, Predicted Product"])]
    data["InChI Match"] = [a in b for a, b in zip(data["InChI, Product"], data["InChI, Predicted Product"])]
    data["InChIKey Match"] = [a in b for a, b in zip(data["InChIKey, Product"], data["InChIKey, Predicted Product"])]

    # Prepare for export
    classvecs_df = pd.DataFrame.from_dict(classvecs, orient="index").drop(columns=[1]).reset_index().rename(columns={"index": "Class", 0:"Average Vector"})

    # Outputs
    run = 1
    while os.path.exists(f"{filepath}/results/{run}"):
        run += 1
    runpath = f"{filepath}/results/{run}"
    os.makedirs(runpath)
    #runpath = f"{filepath}/results/{run}"

    classvecs_df.to_pickle(f"{runpath}/classvecs.pkl")
    data.to_pickle(f"{runpath}/results.pkl")
    #data.to_csv(f"{runpath}/results.csv", index=False)

    end = datetime.now()
    tdiff = end - start

    config['postrun'] = {'run': run, 'jobtime': str(tdiff)}

    with open(f'{runpath}/config.ini', 'w') as configfile:    # save
        config.write(configfile)

    with open(f"{runpath}/log.txt", "w") as logfile:
        logfile.write(f"Parameters for Run {run}\n")
        logfile.write("-------------------------\n")
        logfile.write(f"Completed on: {end}\n")
        logfile.write("=====================================\n")
        logfile.write(f"Maximum SMILES Length:\t{params['max_smiles_length']}\n")
        logfile.write(f"Minimum Class Size:\t{params['min_class_size']}\n")
        logfile.write(f"Random State:\t{params['random_state']}\n")
        logfile.write(f"Combined Classes for Avg Vec?:\t{params['combine']}\n")
        logfile.write(f"Beamsearch Parameter:\t{params['k']}\n")
        logfile.write(f"Canonicalization Engine:\t{params['engine']}\n")
        logfile.write(f"Iterations:\t{params['iterations']}\n")
        if params['other_notes'] != False:
            logfile.write(f"\nOther Notes:\t{params['other_notes']}")
        logfile.write(f"Total Job Run Time (h): {tdiff}")
