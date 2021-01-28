import xml.etree.ElementTree as ET
import xmltodict
import re


def _xmltodict(file_path):
    """
    Load xml file, parse it and transform it to a python dictionary
    :param file_path: path to the file
    :return: dictoanry that corresponds to tree structure of xml file
    """

    tree = ET.parse(file_path)
    root = tree.getroot()
    xmlstr = ET.tostring(root)
    xml_tree = xmltodict.parse(xmlstr)

    return xml_tree


def _re_xml(file_path, keyword):
    """

    :param file_path:
    :param keyword:
    :return:
    """

    relevant_strings = []
    with open(file_path, "r+", encoding='utf-8') as file:
        while True:
            next_line = file.readline()
            if re.search(r'(?=.*<abstract)(?=.*lang="eng").*', next_line):
                next_line = file.readline()
                while not re.search(f'</abstract>', next_line):
                    relevant_strings.append(next_line)
                    next_line = file.readline()
                break

    super_string = ""
    for string in relevant_strings:
        sub_string = re.search(r'.*?\>(.*)<.*', string)
        if super_string:
            super_string += sub_string.group(1)
    return super_string
