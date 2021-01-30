
"""Adapted from: https://gist.github.com/dschwertfeger/3288e8e1a2d189e5565cc43bb04169a1"""

import math
import re
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import os
from Utilities.directories import data
import pathlib

_SEED = 2020
_COMPRESSION_SCALING_FACTOR = 4
_COMPRESSION_LIB = "ZLIB"  # 'ZLIB is the coompression type


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _parallelize(func, data):
    processes = cpu_count() - 1
    with Pool(processes) as pool:
        # We need the enclosing list statement to wait for the iterator to end
        # https://stackoverflow.com/a/45276885/1663506
        list(tqdm(pool.imap_unordered(func, data), total=len(data)))


class TFRecordsConverter:
    """Convert XML files to TFRecords."""

    # When compression is used, resulting TFRecord files are four to five times
    # smaller. So, we can reduce the number of shards by this factor
    _COMPRESSION_SCALING_FACTOR = 4

    def __init__(self, filepaths, output_dir, test_size, val_size):
        """

        :param filepaths: pandas dataframe with filepaths
        :param output_dir:
        :param test_size: in percent of the complete dataset
        :param val_size: in percent of the complete dataset
        """
        self.output_dir = output_dir

        # Shuffle data by "sampling" the entire data frame
        self.filepaths = filepaths.sample(frac=1, random_state=_SEED)

        # Calculate number of instances for each sub dataset
        n_samples = len(filepaths)
        self.n_test = math.ceil(test_size * n_samples)
        self.n_val = math.ceil(val_size * n_samples)
        self.n_train = n_samples - self.n_test - self.n_val

        # Determine number of shards per sub dataset
        self.n_shards_test = self._n_shards(self.n_test)
        self.n_shards_val = self._n_shards(self.n_val)
        self.n_shards_train = self._n_shards(self.n_train)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def __repr__(self):
        return ('{}.{}(output_dir={}, n_shards_train={}, n_shards_test={}, '
                'n_shards_val={}, n_train={}, '
                'n_test={}, n_val={})').format(
            self.__class__.__module__,
            self.__class__.__name__,
            self.output_dir,
            self.n_shards_train,
            self.n_shards_test,
            self.n_shards_val,
            self.n_train,
            self.n_test,
            self.n_val,
        )

    def _n_shards(self, n_samples):
        """
        Compute number of shards for number of samples.

        TFRecords are split into multiple shards. Each shard's size should be
        between 100 MB and 200 MB according to the TensorFlow documentation.

        :param
        n_samples: int
            The number of samples to split into TFRecord shards.
        :return:
        n_shards: int
            The number of shards needed to fit the provided number of samples.
        """
        shard_size = 2 * (10**8) # 200 mb maximum
        avg_file_size = 600  # rough estimation since the file size per document varies a lot.
        files_per_shard = math.ceil(shard_size / avg_file_size) * _COMPRESSION_SCALING_FACTOR
        return math.ceil(n_samples / files_per_shard)

    def _process_files(self, shard_data):
        """
        Write TFRecord file.

        :param
        shard_data: tuple(str, list)
            A tupöe containing the shard path and the list of indices to write to it.
        :return:
        """

        shard_path, indices = shard_data
        with tf.io.TFRecordWriter(shard_path, options=_COMPRESSION_LIB) as out:
            for index in indices:

                row = self.filepaths.iloc[index, :]  # get the respective filepath

                # Extract features
                # abstract
                abstract_text = self.parser(row["path"])

                # Extract label
                label = row["level1codes"] # get label

                example = tf.train.Example(features=tf.train.Features(feature={
                    'abstract': _bytes_feature(abstract_text.encode()),
                    'label': _int64_feature(label),
                }))

                out.write(example.SerializeToString())

    def _get_shard_path(self, split, shard_id, shard_size):
        """
        Construct a shard file path.
        :param split: str
            Split into train, test, validate
        :param shard_id: int
            shard id
        :param shard_size: int
            number of samples this shard contains
        :return:
            shard path: str
        """

        return os.path.join(self.output_dir, f"{split}-{shard_id}-{shard_size}.tfrec")

    def _split_data_into_shards(self):
        """
        Split data into train, test, validate set. Then divide each data set into the specified number of TFRecords
        shards.
        :return: list [tuple]
             Each int this list is a tuple which contains the shard
             path and a list of indices to write to it.
        """

        shards = []

        splits = ('train', 'test', 'validate')
        split_sizes = (self.n_train, self.n_test, self.n_val)
        split_n_shards = (self.n_shards_train, self.n_shards_test, self.n_shards_val)

        offset = 0
        for split, size, n_shards in zip(splits, split_sizes, split_n_shards):
            print('Splitting {} set into TFRecord shards...'.format(split))
            shard_size = math.ceil(size / n_shards)
            cumulative_size = offset + size
            for shard_id in range(1, n_shards + 1):
                step_size = min(shard_size, cumulative_size - offset)
                shard_path = self._get_shard_path(split, shard_id, step_size)

                file_indices = np.arange(offset, offset + step_size)
                shards.append((shard_path, file_indices))
                offset += step_size

        return shards

    def convert(self):
        """Convert to TFRecords."""

        shard_splits = self._split_data_into_shards()
        _parallelize(self._process_files, shard_splits)

        print('Number of training examples: {}'.format(self.n_train))
        print('Number of testing examples: {}'.format(self.n_test))
        print('Number of validation examples: {}'.format(self.n_val))
        print('TFRecord files saved to {}'.format(self.output_dir))

    @staticmethod
    def parser(path):
        """

        :param path:
        :return:
        """
        relevant_strings = []
        with open(path, "r+", encoding='utf-8') as file:
            while True:
                next_line = file.readline()
                if re.search(r'(?=.*<abstract)(?=.*lang="eng").*', next_line):
                    next_line = file.readline()
                    while not re.search(r'</abstract>', next_line):
                        relevant_strings.append(next_line)
                        next_line = file.readline()
                    break

        super_string = ""
        for string in relevant_strings:
            sub_string = re.search(r'.*?\>(.*)<.*', string)
            if sub_string:
                super_string += sub_string.group(1)
        return super_string


# Convert to tfRecord
if __name__ == '__main__':
    labels = pd.read_csv("level1_labels.csv", index_col=0)
    features = pd.read_csv("../../Utilities/statistics.csv", index_col=0)
    patent_data = pd.concat([features, labels], axis=1)

    # %%
    patent_data = patent_data[patent_data["level1labels"].notna()]  # drop unlabeled patents
    patent_data = patent_data[patent_data["abstract"] == 1]  # drop patents that don't contain an abstract
    #print(f"Number of examples: {labels.size}")
    #print(patent_data["level1labels"].value_counts())

    # %%
    # drop AI because of the small number of instances
    patent_data = patent_data[patent_data["level1labels"] != "Artificial Intelligence (AI)"]

    # %%
    # convert labels to categorical and create integer codes
    patent_data["level1labels"] = pd.Categorical(patent_data["level1labels"])
    patent_data["level1codes"] = patent_data["level1labels"].cat.codes

    output_dir = pathlib.Path.joinpath(data, "1.Abstract-SingleClass")
    converter = TFRecordsConverter(patent_data, output_dir, 0.1, 0.1)
    converter.convert()



