import pandas as pd
import re
import numpy as np

labels = pd.read_csv("ip7-data-updated.csv", index_col=0)
all_classes = []
for l in labels["technology-field"]:
    for ele in l.split(","):
        ele = re.sub(r'^\s+|\s+$|"', '', ele)
        all_classes.append(ele)
all_classes = pd.DataFrame(all_classes)
all_classes = all_classes.loc[:, 0].unique()
all_classes = np.sort(all_classes)
np.savetxt("all-classes.txt", all_classes, fmt='%s', delimiter='\n')
print(all_classes)