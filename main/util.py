import os
import subprocess
import sys
import numpy as np
import pandas as pd


def vecPredictProba(models, X):
    if type(X) is list:
        X = np.array(X)

    probas = np.empty((X.shape[0], len(models)))
    for i, model in enumerate(models):
        probas[:, i] = model.predict_proba(X)[:, 1]
    return probas


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def evaluateDataset(dataset, name):
    os.chdir(sys.path[0])
    os.chdir("../MDP_Dataset_Builder")
    np.save("./starting_combinations.npy", dataset)

    if os.name == "posix":
        venv = "source ../venv/bin/activate &&"    # Adjust path to project venv ("" if no venv)
        command = f"{venv} ./evaluate_adaptations.sh ./starting_combinations.npy"
        subprocess.run(command, shell=True, executable="/bin/bash")
    else:
        venv = "..\\venv\\Scripts\\activate &&"    # Adjust path to project venv ("" if no venv)
        command = f"{venv} evaluate_adaptations.bat ./starting_combinations.npy"
        subprocess.run(command, shell=True)

    # Rename the file
    os.chdir("..")
    source_file = './MDP_Dataset_Builder/merge.csv'
    new_file = './results/' + name + '.csv'
    if os.path.exists(new_file):
        os.remove(new_file)
    # i = 1
    # while os.path.exists(new_file):
    #     new_file = './results/' + name + "(" + i + ")" + '.csv'
    os.rename(source_file, new_file)


# def evaluateAdaptations(dataset_anchors, dataset_custom, dataset_SHAP, dataset_FI, dataset_fitest, dataset_random, dataset_nsga,  featureNames):
def evaluateAdaptations(dataset_anchors, dataset_custom, dataset_nsga,  featureNames):
    AnchorsAdaptations = pd.DataFrame(dataset_anchors['custom_adaptation'].to_list(), columns=featureNames)
    customAdaptations = pd.DataFrame(dataset_custom['custom_adaptation'].to_list(), columns=featureNames)
    # SHAPAdaptations = pd.DataFrame(dataset_SHAP['custom_adaptation'].to_list(), columns=featureNames)
    # FIAdaptations = pd.DataFrame(dataset_FI['custom_adaptation'].to_list(), columns=featureNames)
    # FitestAdaptations = pd.DataFrame(dataset_fitest['custom_adaptation'].to_list(), columns=featureNames)
    # RandomAdaptations = pd.DataFrame(dataset_random['custom_adaptation'].to_list(), columns=featureNames)
    NSGAadaptations = pd.DataFrame(dataset_nsga['custom_adaptation'].to_list(), columns=featureNames)

    evaluateDataset(AnchorsAdaptations, "anchorsDataset")
    evaluateDataset(customAdaptations, "customDataset")
    # evaluateDataset(SHAPAdaptations, "SHAPDataset")
    # evaluateDataset(FIAdaptations, "FIDataset")
    # evaluateDataset(FitestAdaptations, "FitestDataset")
    # evaluateDataset(RandomAdaptations, "RandomDataset")
    evaluateDataset(NSGAadaptations, "NSGADataset")




def readFromCsv(path):
    results = pd.read_csv(path)
    columns = ["custom_adaptation", "custom_confidence"]

    # numpy arrays are read as strings, must convert them back in arrays
    for c in columns:
        results[c] = results[c].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=' '))

    return results
