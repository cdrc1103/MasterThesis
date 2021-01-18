import xml.etree.ElementTree as ET
import xmltodict
from os import listdir


def load_xml(file_dir):
    """
    Loads all xml files from the directory and converts them to a list of
    python dictionaries
    :param file_dir: Directory to the files
    :return: list of python dictionaries
    """

    file_names = [f for f in listdir(file_dir)]
    dict_list = []
    for name in file_names:
        tree = ET.parse(file_dir + "/" + name)
        root = tree.getroot()
        xmlstr = ET.tostring(root)
        dict_list.append(xmltodict.parse(xmlstr))

    return dict_list
