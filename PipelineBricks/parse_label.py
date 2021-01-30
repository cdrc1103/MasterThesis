

"""
Experiment 1 Abstract BertBaseUncased SingleClass

Parse label hierarchy and pick all level 1 classes that are assigned to a document.
If there are several level 1 classes pick a random one and save in dataframe.
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import random

ip7data = pd.read_csv("../../Utilities/ip7-data-updated.csv", index_col=0)
statistics = pd.read_csv("../../Utilities/feature_stats.csv", index_col=0)

level1_labels_list = []
for doc_nr, label_available in zip(statistics.index, tqdm(statistics['label'])): # iterate over rows of df
    if label_available: # if the respective document is in the statistics file
        label_hierarchy = ip7data.loc[doc_nr, 'technology-field'] # get the row of the respective file number in the statistics file
        level1_labels = set()
        for ele in label_hierarchy.split(','): # split all classes
            class_list = [re.sub(r'^\s+|\s+$|"', '', h) for h in ele.split('/')] # split all hierarchies within respective class
            # if not class_list[0] in unwanted_classes: # skip this one because it has no information
            level1_labels.add(class_list[0]) # add highest level of hierarchy to set so that there are no duplicates
        if level1_labels: # if there is a label in the set
            random_label = random.sample(level1_labels, 1)[0]
            level1_labels_list.append(random_label)
        else:
            level1_labels_list.append(0)
    else:
        level1_labels_list.append(0)
labels = pd.DataFrame(level1_labels_list, index=statistics.index, columns=["level1labels"])
labels = labels.replace(0, np.nan)
labels.to_csv("ex1_labels.csv")