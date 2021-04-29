import pandas as pd
import re
import numpy as np

"""
Creates unique set of the classes.
"""

labels = pd.read_csv("ip7-data-updated.csv", index_col=0)
all_classes = []
for l in labels["technology-field"]:
    for ele in l.split(","):
        ele = re.sub(r'^\s+|\s+$|"', '', ele)
        all_classes.append(ele)
all_classes = pd.DataFrame(all_classes)
all_unique_classes = all_classes.loc[:, 0].unique()
n_unique_classes = all_classes.loc[:,0].value_counts()
stats = pd.DataFrame({"label path": all_unique_classes, "counts": n_unique_classes.to_list()})
stats.to_csv("all_classes.csv")