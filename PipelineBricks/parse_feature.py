import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langdetect import detect


def parse(features, path, patent_id):
    """

    :param path_df:
    :return:
    """

    tree = ET.parse(path)
    root = tree.getroot()

    parsing_results = {}

    if "abstract" in features:
        text = root.find('.//abstract[@lang="eng"]/').text
        if detect(text) == "en":
            parsing_results["abstract"] =text

    return parsing_results, patent_id


def process_files(feature_stats, feature_list):
    """

    :param path_list:
    :return:
    """

    dataset = pd.DataFrame(index=feature_stats.index)
    dataset[feature_list] = None

    with ThreadPoolExecutor() as executor:
        futures = []
        for path, patent_id in zip(feature_stats["path"], feature_stats.index):
            futures.append(executor.submit(parse, feature_list, path, patent_id))

        for future in tqdm(as_completed(futures), total=len(feature_stats)):
            text, patent_id = future.result()
            for f in feature_list:
                dataset.loc[patent_id, f] = text[f]

    return dataset
