import pandas as pd
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor

_KEYWORD = "abstract"


def re_search(file_path):
    """

    :param file_path:
    :param keyword:
    :return:
    """

    relevant_strings = []
    with open(file_path, "r+", encoding='utf-8') as file:
        all_lines = [line for line in file]
        n_lines = len(all_lines)
        i = 0
        while i < n_lines:
            next_line = all_lines[i]
            if re.search(r'(?=.*<abstract)(?=.*lang="eng").*', next_line):
                i+=1
                next_line = all_lines[i]
                while not re.search(r'</abstract>', next_line):
                    relevant_strings.append(next_line)
                    i+=1
                    next_line = all_lines[i]
                break
            i+=1


    super_string = ""
    for string in relevant_strings:
        sub_string = re.search(r'.*?\>(.*)<.*', string)
        if sub_string:
            super_string += sub_string.group(1)
    if super_string:
        return super_string
    else:
        return None


def process_files():
    """

    :param path_list:
    :return:
    """

    labels = pd.read_csv("ex1_labels.csv", index_col=0)
    features = pd.read_csv("../../Utilities/statistics.csv", index_col=0)
    patent_data = pd.concat([features, labels], axis=1)

    patent_data = patent_data[patent_data["level1labels"].notna()]  # drop unlabeled patents
    patent_data = patent_data[patent_data["abstract"] == 1]  # drop patents that don't contain an abstract
    # print(f"Number of examples: {labels.size}")
    # print(patent_data["level1labels"].value_counts())

    # drop AI because of the small number of instances
    patent_data = patent_data[patent_data["level1labels"] != "Artificial Intelligence (AI)"]

    # convert labels to categorical and create integer codes
    patent_data["level1labels"] = pd.Categorical(patent_data["level1labels"])
    patent_data["level1codes"] = patent_data["level1labels"].cat.codes

    patent_data = patent_data.sample(frac=1, random_state=10000) # shuffle data set

    data_set = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, abstract in enumerate(tqdm(executor.map(re_search, patent_data["path"]), total=len(patent_data))):
            patent_id = patent_data.index[i]
            if abstract:
                data_set.append((patent_id, abstract))
    # for patent_id, path in zip(patent_data.index, tqdm(patent_data["path"])):
    #     print(f"patent_id: {patent_id}")
    #     abstract_text = re_search(path)
    return data_set


if __name__ == "__main__":
    data_set = process_files()
    data_set = pd.DataFrame(data_set, columns=["patent_id", "abstract"])
    data_set = data_set.set_index("patent_id")
    data_set.to_csv("dataset.csv")

