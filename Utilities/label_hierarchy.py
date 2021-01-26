import csv
import re

def string2dict(string):
    pass

with open("ip7-data-updated.csv") as file:
    data = csv.reader(file, delimiter=",")

    data_tree = {}
    # for row in data:
    #     # labels = str(row[1])
    #
    #
    #     #data_tree[row[0]] =

    test = '"Active ingredients/Z. monitorings Actives, Cleansing, Hair care/shampoo, Hair care/conditioner"'
    classes = []

    level1_labels = set()
    for ele in test.split(','):
        class_list = [re.sub(r'^\s+|\s+$|"', '', h) for h in ele.split('/')]
        level1_labels.add(class_list[0])



