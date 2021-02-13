import pandas as pd


def create_path_dataset(dataset_dir):
    """
    Reads all file paths in dataset_dir and packs them in a .csv file
    :param dataset_dir: directory of the dataset, should be a pathlib object
    """

    # Combine root directory and filename to a path and put the in a tf data set
    path_list = dataset_dir.glob("*xml")
    df = pd.DataFrame(path_list, columns=["path"])
    df.to_csv("paths.csv", index=False)

