import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm
import pathlib
import pandas as pd
import xml.etree.ElementTree as ET

FEATURE_LIST = ["abstract",  "description", "claims", "title"]


def search_features(patent_path, patent_id, label):

    occurrences = np.zeros([len(FEATURE_LIST)], dtype=bool)  # 3 feature occurences are checked

    if label:
        tree = ET.parse(patent_path)
        root = tree.getroot()

        # check feature occurrence
        for i, feature in enumerate(FEATURE_LIST):
            if root.find('.//%s[@lang="eng"]/' % feature) is not None:
                occurrences[i] = True

    return occurrences, patent_id, patent_path


def feature_occurences(path_dataset, label_dataset):

    print("Checking features...")

    path_list = pd.read_csv(path_dataset, index_col=0).iloc[:, 0].to_list()  # read all file paths
    label_dataset = pd.read_csv(label_dataset, index_col=0)
    occurrences = np.zeros([len(path_list), len(FEATURE_LIST)], dtype=bool)

    new_id_order = []
    new_path_list = []

    occurrence_sum = np.zeros([len(FEATURE_LIST)])

    with ThreadPoolExecutor() as executor:
        futures = []
        for path, patent_id, label in zip(path_list, label_dataset.index, label_dataset.iloc[:, 0]):
            futures.append(executor.submit(search_features, path, patent_id, label))

        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(path_list))):
            occ, patent_id, patent_path = future.result()
            occurrences[i,:] = occ
            new_id_order.append(patent_id)
            new_path_list.append(patent_path)
            occurrence_sum += occ

    relative_occuraences = np.round(occurrence_sum / len(path_list), 2)
    relative_occurences_dict = {f: o for f, o in zip(FEATURE_LIST, relative_occurences)}

    stat_df = pd.DataFrame(index=new_id_order)
    stat_df["path"] = new_path_list
    stat_df[FEATURE_LIST] = occurrences
    stat_df.sort_index(inplace=True)
    label_dataset.sort_index(inplace=True)
    stat_df = pd.concat([stat_df, label_dataset], axis=1)

    return relative_occurences_dict, stat_df


def label_occurences(path_dataset, ip7label_hierarchy):

    print("Checking labels...")
    path_list = pd.read_csv(path_dataset, index_col=0).iloc[:, 0].to_list()  # read all file paths
    ip7labels = pd.read_csv(ip7label_hierarchy, index_col=0)

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

