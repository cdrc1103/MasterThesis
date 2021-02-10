import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

length_threshold = 40

def get_text(node):
    return ((node.text or '') +
            ''.join(map(get_text, node)) +
            (node.tail or ''))


def clean_text(text):
    text = " ".join(text.split()) # remove multiples of white / tab spaces
    text =text.replace("\n", "") # remove newline characters
    if len(text) <= length_threshold: return None # if the text is to short return nothing instead
    return text


def get_abstract(root):
    result = root.find('.//abstract[@lang="eng"]/*')
    text = get_text(result)
    return clean_text(text)


def get_title(root):
    text = root.find('bibliographic-data/invention-title[@lang="eng"]').text or ""
    return clean_text(text)


ops_map = {
    "abstract": get_abstract,
    "title": get_title
}


def parse(features, path, patent_id):
    """

    :param path_df:
    :return:
    """

    tree = ET.parse(path)
    root = tree.getroot()

    parsing_results = {}

    for f in features:
        parsing_results[f] = ops_map[f](root)

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
            results, patent_id = future.result()
            for f in feature_list:
                dataset.loc[patent_id, f] = results[f]

    return dataset
