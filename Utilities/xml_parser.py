import xml.etree.ElementTree as ET
import xmltodict
from os import listdir


def parse_xml(file_paths):
    """
    Loads all xml files from the directory and converts them to a list of
    python dictionaries
    :param file_paths: path to the file or list of file paths
    :return: list of python dictionaries
    """

    if not type(file_paths) == list:
        file_paths = [file_paths]
    dict_list = []
    for path in file_paths:
        tree = ET.parse(path)
        root = tree.getroot()
        xmlstr = ET.tostring(root)
        dict_list.append(xmltodict.parse(xmlstr))

    return dict_list
