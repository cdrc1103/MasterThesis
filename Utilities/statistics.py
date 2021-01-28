import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import re
import pathlib
import pandas as pd
from Utilities.directories import lexis_data

feature_list = ["claim", "description", "abstract", "citation"]


def search_features(file):
    """
    Searches for features on the first level of the dictionary and check if their text is in english.
    :param file: path to the file
    :return:
    """

    with open(file, encoding='utf-8') as patent:

        string = patent.read() # read file

        occurrences = np.zeros([len(feature_list)], dtype=bool)  # 4 feature occurences are checked
        # check feature occurrence
        for i, feature in enumerate(feature_list):
            if re.search(f'<{feature}', string):
                occurrences[i] = 1
    return occurrences


def feature_occurences(path_dataset):
    """
    Count the occurrence of features among all documents that can be found in the text.
    :param
        path_dataset: path to file that contains all filepaths to documents
        feature_list: list of features that should be checked
    """

    path_list = pd.read_csv(path_dataset)['paths'].to_list()  # read all file paths

    occurrences = np.zeros([len(path_list), len(feature_list)], dtype=bool)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, occ in enumerate(tqdm(executor.map(search_features, path_list), total=len(path_list))):
            occurrences[i, :] = occ

        occurence_sum = np.sum(occurrences, 0)
        relative_occurences = np.round(occurence_sum / len(path_list), 2)
        relative_occurences_dict = {f: o for f, o in zip(feature_list, relative_occurences)}

        patent_id = [pathlib.Path(p).stem for p in path_list]
        stat_df = pd.DataFrame(index=patent_id)
        stat_df["path"] = path_list
        stat_df[feature_list] = occurrences

        return relative_occurences_dict, stat_df


def label_occurences(ip7data, path_dataset):

    path_list = pd.read_csv(path_dataset)['paths'].to_list()  # read all file paths
    ip7labels = pd.read_csv(ip7data, index_col=0)

    # check label occurence
    occurrences = np.zeros([len(path_list)], dtype=bool)
    patent_id = [pathlib.Path(p).stem for p in path_list]
    for i, path in enumerate(path_list):
        if patent_id[i] in ip7labels.index:
            occurrences[i] = 1

    relative_occurrences_dict = {"label": np.round(occurrences.sum() / len(path_list))}
    stat_df = pd.DataFrame(index=patent_id)
    stat_df["labeled"] = occurrences

    return relative_occurrences_dict, stat_df


# Check how many of the patents have an english set of claim, description, citation and abstract
# The Python multiprocessing style guide recommends to place the multiprocessing code inside the
# __name__ == '__main__' idiom. This is due to the way the processes are created on Windows.
# The guard is to prevent the endless loop of process generations.

if __name__ == '__main__':
    path_dataset_dir = pathlib.Path.joinpath(lexis_data, "paths.csv")
    ip7data = pathlib.Path.joinpath(lexis_data, "ip7-data-updated.csv")
    _, df1 = feature_occurences(path_dataset_dir)
    _, df2 = label_occurences(ip7data, path_dataset_dir)
    df3 = pd.concat([df1, df2], axis=1)
    df3.to_csv("statistics.csv")
