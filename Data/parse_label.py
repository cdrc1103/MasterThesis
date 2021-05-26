"""
Parse patent labels
"""

from tqdm import tqdm
import pandas as pd
import numpy as np
import re

# Name for irrelevant classes
IRRELEVANT = "Irrelevant"


def parse_label(ip7label_hierarchy, label_occ):
    """
    Parse label hierarchy and pick all level 1 classes that are assigned to a document.
    """
    ip7data = pd.read_csv(ip7label_hierarchy, index_col=0)
    label_occ = pd.read_csv(label_occ, index_col=0)
    label_colname = ip7data.columns[0]
    level1_labels_list = []
    for doc_nr, label_available in zip(label_occ.index, tqdm(label_occ.iloc[:, 0])): # iterate over rows of df
        if label_available: # if the respective document is in the statistics file
            label_hierarchy = ip7data.loc[doc_nr, label_colname] # get the row of the respective file number in the statistics file
            level1_labels = set()
            for ele in label_hierarchy.split(','): # split all classes
                class_list = [re.sub(r'^\s+|\s+$|"', '', h) for h in ele.split('/')] # split all hierarchies within respective class
                level1_labels.add(class_list[0]) # add highest level of hierarchy to set so that there are no duplicates
            if level1_labels: # if there is a label in the set
                level1_labels_list.append(list(level1_labels))
            else:
                level1_labels_list.append(np.nan)
        else:
            level1_labels_list.append([IRRELEVANT])
    labels = pd.Series(level1_labels_list, index=label_occ.index, name="level1labels")
    return labels
