import re
import pandas as pd


class Statistics:
    def __init__(self):
        self.abstract = 0

    def calc_statistics(self, path_dataset):

        path_list = pd.read_csv(path_dataset)['paths'].to_list()
        n_file = len(path_list)

        for i in range(n_file):
            with open(path_list[i], encoding='utf-8') as file:
                if re.search("<claims", file.read()):
                    self.abstract += 1
                if i % 100 == 0:
                    print(f"{i} of {n_file}")
        print(f"abstract availability: {self.abstract / n_file}")


dir = r"E:\MLData\thesis\Datasets\automated-classification-data\automated-classification-data\lexisnexis-data\LexisNexis\paths.csv"
s = Statistics()
s.calc_statistics(dir)
