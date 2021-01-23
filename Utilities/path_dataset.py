import pandas as pd
from Utilities.directories import *
import pathlib


def create(directory):
    """
    Loads all file path in dir_name and packs them in a dataset
    :param dataset_name: Name of the dataset directory. Should be in the root directory.
    """
    root = pathlib.Path(directory)
    # Combine root directory and filename to a path and put the in a tf data set
    path_list = root.glob("*xml")
    df = pd.DataFrame(path_list, columns=["paths"])
    df.to_csv(f"{root}/paths.csv", index=False)


create(r"E:\MLData\thesis\Datasets\automated-classification-data\automated-classification-data\lexisnexis-data\LexisNexis")
