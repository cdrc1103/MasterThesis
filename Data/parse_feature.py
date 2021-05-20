# Parsing
import xml.etree.ElementTree as ET

# Data processing
import pandas as pd
import numpy as np
from datetime import datetime

# Progress bar
from tqdm import tqdm

# Multithreading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Database
from secrets import USERNAME, PASSWORD
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.postgresql import *
from sqlalchemy.types import TEXT, ARRAY, DATE
engine = create_engine(f'postgresql+psycopg2://{USERNAME}:{PASSWORD}@localhost:5432/Thesis', echo=False)

dtypes = {
    'level1labels': ARRAY(TEXT),
    'claim': ARRAY(TEXT),
    'patentid': VARCHAR(100),
    'abstract': TEXT,
    'title': TEXT,
    'description': TEXT,
    "section": ARRAY(TEXT),
    "class": ARRAY(TEXT),
    "subclass": ARRAY(TEXT),
    "main-group": ARRAY(TEXT),
    "subgroup": ARRAY(TEXT),
    "date": DATE
}

upload_freq = 5000 # frequency of uploads to database


def get_text(node):
    if node is None:
        return ""
    return ((node.text or '') +
            ''.join(map(get_text, node)) +
            (node.tail or ''))


def clean_text(text, length_threshold=40):
    text = " ".join(text.split()) # remove multiples of white / tab spaces
    text =text.replace("\n", "") # remove newline characters
    if len(text) <= length_threshold: return np.nan # if the text is to short return nothing instead
    return text


def get_abstract(root):
    result = root.find('.//abstract[@lang="eng"]/*')
    text = get_text(result)
    return clean_text(text)


def get_title(root):
    result = root.find('bibliographic-data/invention-title[@lang="eng"]')
    text = get_text(result)
    return clean_text(text, 0)


def get_claim(root):
    text = root.findall('.//claims[@lang="eng"]/claim/')
    text = [get_text(t) for t in text]
    new_text = []
    if text:
        for t in text:
            new_t = clean_text(t, 0)
            if isinstance(new_t, str):
                new_text.append(new_t)
        return new_text
    else:
        return np.nan


def get_description(root):
    text = root.findall('.//description[@lang="eng"]/')
    text = " ".join([get_text(t) for t in text])
    return clean_text(text)


def get_cpc(root):
    cpc_elements = ["section", "class", "subclass", "main-group", "subgroup"]
    cpc_dict = {}
    for ele in cpc_elements:
        values = root.findall(f'bibliographic-data/classifications-cpc/classification-cpc/{ele}')
        cpc_dict[ele] = set(map(lambda v: v.text, values)) or np.nan
    return pd.Series(cpc_dict).astype('object')


def get_date(root):
    result = root.findall('.//priority-claims/priority-claim/date')
    date_raw = [int(r.text) for r in result]
    if date_raw:
        date = min([datetime.strptime(str(d), "%Y%m%d") for d in date_raw])
        return date.date()
    else:
        return np.nan


ops_map = {
    "abstract": get_abstract,
    "title": get_title,
    "claim": get_claim,
    "description": get_description,
    "cpc": get_cpc,
    "date": get_date
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


def process_files(feature_occ, feature_list, table_name):
    """

    :param path_list:
    :return:
    """

    with ThreadPoolExecutor() as executor:
        futures = []
        for path, patent_id in zip(feature_occ["path"], feature_occ.index):
            futures.append(executor.submit(parse, feature_list, path, patent_id))

        counter = 1
        n_rows = len(feature_occ)
        temp_dict = {f: [] for f in feature_list}
        id_list = []
        for future in tqdm(as_completed(futures), total=n_rows):
            result, patent_id = future.result()
            for f in feature_list:
                temp_dict[f].append(result[f])
            id_list.append(patent_id)
            if counter % upload_freq == 0 or counter == n_rows:
                for f in feature_list:
                    temp_df = pd.DataFrame(temp_dict[f], index=id_list, columns=[f])
                    temp_df["level1labels"] = feature_occ["label"]
                    temp_df.index.rename("patentid", inplace=True)
                    with engine.begin() as conn:
                        temp_df = temp_df.dropna()
                        temp_df.to_sql(
                            name=table_name,
                            con=conn,
                            if_exists='append',
                            dtype=dtypes,
                            method='multi')
                temp_dict = {f: [] for f in feature_list}
                id_list = []
            counter += 1
        return temp_df