import zipfile
import os
import pathlib


class Unzip:

    def __init__(self, dataset_dir):
        """
        Unzip all zip files in dataset_dir and check if a the unzipped file already exists before writing
        it to the directory
        :param dataset_dir: directory to the zip files
        """

        self.target_location = dataset_dir
        self.target_zip = self.target_location.glob("*.zip")

    def check(self, member):
        if os.path.isfile(pathlib.Path.joinpath(self.target_location, member)):
            pass
        else:
            self.zip_file.extract(member, self.target_location)

    def unzip(self):
        for i, path in enumerate(self.target_zip):
            with zipfile.ZipFile(path) as self.zip_file:
                file_list = list(self.zip_file.namelist())
                for file in file_list:
                    self.check(file)
            print(f"file: {i}")
