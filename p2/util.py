import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
import csv
import itertools

# these are the fifteen malware classes we're looking for
malware_classes = ["Agent", "AutoRun", "FraudLoad", "FraudPack", "Hupigon", "Krap",
                   "Lipler", "Magania", "None", "Poison", "Swizzor", "Tdss",
                   "VB", "Virut", "Zbot"]

# a function for writing predictions in the required format
def write_predictions(predictions, ids, outfile):
    """
        assumes len(predictions) == len(ids), and that predictions[i] is the
        index of the predicted class with the malware_classes list above for
        the executable corresponding to ids[i].
        outfile will be overwritten
        """
    with open(outfile,"w+") as f:
        # write header
        f.write("Id,Prediction\n")
        for i, history_id in enumerate(ids):
            f.write("%s,%d\n" % (history_id, predictions[i]))

def get_syscall_names(direc="train"):
    fds = [] # list of feature dicts
    classes = []
    ids = []
    calls_dict = {}
    i = 1
    for datafile in os.listdir(direc):
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(malware_classes.index(clazz))
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features

        target_elements = tree.findall('.//all_section/*')

        for el in target_elements:
            if el.tag not in calls_dict:
                calls_dict[el.tag] = i
                i += 1
        
    print len(calls_dict)
    with open('systemcalls.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in calls_dict.items():
            writer.writerow([key, value])


def dict_from_csv(direc):
    dic = {}
    sys_map = csv.reader(open(direc, 'r'))
    for row in sys_map:
        dic[str(row[0])] = row[1]
    return dic

def create_syscall_ngrams(n):
    syscalls = dict_from_csv("systemcalls.csv").keys()
    ngrams = itertools.combinations(syscalls, n)
    filename = "syscalls_{}gram.csv".format(n)
    with open(filename, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for f, s in ngrams:
            key = f+"+"+s
            writer.writerow([key, 0])
            key = s+"+"+f
            writer.writerow([key, 0])
        for syscall in syscalls:
            key = syscall+"+"+syscall
            writer.writerow([key, 0])
