import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm
import pathlib
import pandas as pd
from Utilities.directories import lexis_data
import xml.etree.ElementTree as ET

feature_list = ["abstract",  "description", "claims"]


def search_features(patent_path, patent_id, label):
    """
    Search for feature occurrence and check if the language is english.
    :param patent_path:
    :param patent_id:
    :param label:
    :return:
    """

    occurrences = np.zeros([len(feature_list)], dtype=bool)  # 4 feature occurences are checked

    if label:
        tree = ET.parse(patent_path)
        root = tree.getroot()

        # check feature occurrence
        for i, feature in enumerate(feature_list):
            if root.find('.//%s[@lang="eng"]/' % feature) is not None:
                occurrences[i] = True

    return occurrences, patent_id, patent_path


def feature_occurences(path_dataset, label_df):
    """
    Count the occurrence of features among all documents that can be found in the text.
    :param
        path_dataset: path to file that contains all filepaths to documents
        feature_list: list of features that should be checked
    """

    print("Checking features...")

    path_list = pd.read_csv(path_dataset)['paths'].to_list()  # read all file paths
    occurrences = np.zeros([len(path_list), len(feature_list)], dtype=bool)

    new_id_order = []
    new_path_list = []

    occurrence_sum = np.zeros([len(feature_list)])

    with ThreadPoolExecutor() as executor:
        futures = []
        for path, patent_id, label in zip(path_list, label_df.index, label_df["label"]):
            futures.append(executor.submit(search_features, path, patent_id, label))

        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(path_list))):
            occ, patent_id, patent_path = future.result()
            occurrences[i,:] = occ
            new_id_order.append(patent_id)
            new_path_list.append(patent_path)
            occurrence_sum += occ

    relative_occurences = np.round(occurrence_sum / len(path_list), 2)
    relative_occurences_dict = {f: o for f, o in zip(feature_list, relative_occurences)}

    stat_df = pd.DataFrame(index=new_id_order)
    stat_df["path"] = new_path_list
    stat_df[feature_list] = occurrences
    stat_df.sort_index(inplace=True)
    label_df.sort_index(inplace=True)
    stat_df = pd.concat([stat_df, label_df], axis=1)

    return relative_occurences_dict, stat_df


def label_occurences(ip7data, path_dataset):
    """

    :param ip7data:
    :param path_dataset:
    :return:
    """

    print("Checking labels...")
    path_list = pd.read_csv(path_dataset)['paths'].to_list()  # read all file paths
    ip7labels = pd.read_csv(ip7data, index_col=0)

    # check label occurence
    occurrences = np.zeros([len(path_list)], dtype=bool)
    patent_id = [pathlib.Path(p).stem for p in path_list]
    for i, path in enumerate(tqdm(path_list)):
        if patent_id[i] in ip7labels.index:
            occurrences[i] = 1

    relative_occurrences_dict = {"label": np.round(occurrences.sum() / len(path_list), 2)}
    stat_df = pd.DataFrame(index=patent_id)
    stat_df["label"] = occurrences

    return relative_occurrences_dict, stat_df


# Check how many of the patents have an english set of claim, description, citation and abstract

# The Python multiprocessing style guide recommends to place the multiprocessing code inside the
# __name__ == '__main__' idiom. This is due to the way the processes are created on Windows.
# The guard is to prevent the endless loop of process generations.

if __name__ == '__main__':
    path_dataset_dir = pathlib.Path.joinpath(lexis_data, "paths.csv")
    ip7data = "ip7-data-updated.csv"
    relative_label_occ, label_df = label_occurences(ip7data, path_dataset_dir)
    print(f"Relative label occurrence: {relative_label_occ}")
    relative_feature_occ, stat_df = feature_occurences(path_dataset_dir, label_df)
    print(f"Relative feature occurrence: {relative_feature_occ}")
    stat_df.to_csv("feature_stats.csv")
