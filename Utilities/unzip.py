import zipfile
import os
import pathlib


class Unzip:

    def __init__(self):
        self.target_location = pathlib.Path(r'E:\MLData\thesis\Datasets\automated-classification-data\automated-classification-data\lexisnexis-data\new')
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


u = Unzip()
u.unzip()