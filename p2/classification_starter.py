## This file provides starter code for extracting features from the xml files and
## for doing some learning.
##
## The basic set-up: 
## ----------------
## main() will run code to extract features, learn, and make predictions.
## 
## extract_feats() is called by main(), and it will iterate through the 
## train/test directories and parse each xml file into an xml.etree.ElementTree, 
## which is a standard python object used to represent an xml file in memory.
## (More information about xml.etree.ElementTree objects can be found here:
## http://docs.python.org/2/library/xml.etree.elementtree.html
## and here: http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/)
## It will then use a series of "feature-functions" that you will write/modify
## in order to extract dictionaries of features from each ElementTree object.
## Finally, it will produce an N x D sparse design matrix containing the union
## of the features contained in the dictionaries produced by your "feature-functions."
## This matrix can then be plugged into your learning algorithm.
##
## The learning and prediction parts of main() are largely left to you, though
## it does contain code that randomly picks class-specific weights and predicts
## the class with the weights that give the highest score. If your prediction
## algorithm involves class-specific weights, you should, of course, learn 
## these class-specific weights in a more intelligent way.
##
## Feature-functions:
## --------------------
## "feature-functions" are functions that take an ElementTree object representing
## an xml file (which contains, among other things, the sequence of system calls a
## piece of potential malware has made), and returns a dictionary mapping feature names to 
## their respective numeric values. 
## For instance, a simple feature-function might map a system call history to the
## dictionary {'first_call-load_image': 1}. This is a boolean feature indicating
## whether the first system call made by the executable was 'load_image'. 
## Real-valued or count-based features can of course also be defined in this way. 
## Because this feature-function will be run over ElementTree objects for each 
## software execution history instance, we will have the (different)
## feature values of this feature for each history, and these values will make up 
## one of the columns in our final design matrix.
## Of course, multiple features can be defined within a single dictionary, and in
## the end all the dictionaries returned by feature functions (for a particular
## training example) will be unioned, so we can collect all the feature values 
## associated with that particular instance.
##
## Two example feature-functions, first_last_system_call_feats() and 
## system_call_count_feats(), are defined below.
## The first of these functions indicates what the first and last system-calls 
## made by an executable are, and the second records the total number of system
## calls made by an executable.
##
## What you need to do:
## --------------------
## 1. Write new feature-functions (or modify the example feature-functions) to
## extract useful features for this prediction task.
## 2. Implement an algorithm to learn from the design matrix produced, and to
## make predictions on unseen data. Naive code for these two steps is provided
## below, and marked by TODOs.
##
## Computational Caveat
## --------------------
## Because the biggest of any of the xml files is only around 35MB, the code below 
## will parse an entire xml file and store it in memory, compute features, and
## then get rid of it before parsing the next one. Storing the biggest of the files 
## in memory should require at most 200MB or so, which should be no problem for
## reasonably modern laptops. If this is too much, however, you can lower the
## memory requirement by using ElementTree.iterparse(), which does parsing in
## a streaming way. See http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/
## for an example. 

import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse

import util
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import svm

# fix random seed for reproducibility
numpy.random.seed(7)

sys_to_int_map = util.dict_from_csv('systemcalls.csv')

def extract_feats(ffs, direc="train", global_feat_dict=None):
    """
    arguments:
      ffs are a list of feature-functions.
      direc is a directory containing xml files (expected to be train or test).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.

    returns: 
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target classes, and a list of system-call-history ids in order 
      of their rows in the design matrix.
      
      Note: the vector of target classes returned will contain the true indices of the
      target classes on the training data, but will contain only -1's on the test
      data
    """
    fds = [] # list of feature dicts
    classes = []
    ids = [] 
    for datafile in os.listdir(direc):
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)
        rowfd = {}
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features
        [rowfd.update(ff(tree)) for ff in ffs]
        fds.append(rowfd)
        
    X,feat_dict = make_design_mat(fds,global_feat_dict)
    return X, feat_dict, np.array(classes), ids

def extract_lstm_feats(ffs, direc="train", global_feat_dict=None):
    seqs = [] # list of feature dicts
    classes = []
    ids = [] 
    max_seq_len = 500
    for datafile in os.listdir(direc):
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features
        seqs.append(sequence_sys_calls(tree))
        
    X = sequence.pad_sequences(seqs, maxlen=max_seq_len)

    classes = np.array(classes)
    Y = np.zeros((len(classes), max(classes)+1))
    Y[np.arange(len(classes)), classes] = 1

    return X, Y, ids

def get_malware_class_syscall_counts():
    classes = []
    counts = []
    direc="train"
    for datafile in os.listdir(direc):
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features
        counts.append(get_syscall_counts(tree))
    total_counts = [{} for _ in range(max(classes) + 1)]
    for i in range(len(counts)):
        for key, value in counts[i].iteritems():
            if key in total_counts[classes[i]]:
                total_counts[classes[i]][key] += 1
            else:
                 total_counts[classes[i]][key] = 1
    plot_freqs(total_counts)
    return total_counts

def plot_freqs(dict_list):
    plt.rcParams['xtick.labelsize'] = 3
    for i, syscalls in enumerate(dict_list):
        fig = plt.figure()
        centers = range(len(syscalls))
        plt.bar(centers, syscalls.values(), align='center', tick_label=syscalls.keys())
        plt.ylabel("Frequency")
        plt.title("{}: {}".format(i, util.malware_classes[i]))
        fig.autofmt_xdate()
        plt.show()
        plt.clf()

def make_design_mat(fds, global_feat_dict=None):
    """
    arguments:
      fds is a list of feature dicts (one for each row).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.
       
    returns: 
        a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds 
    """
    if global_feat_dict is None:
        all_feats = set()
        [all_feats.update(fd.keys()) for fd in fds]
        feat_dict = dict([(feat, i) for i, feat in enumerate(sorted(all_feats))])
    else:
        feat_dict = global_feat_dict
    cols = []
    rows = []
    data = []        
    for i in xrange(len(fds)):
        temp_cols = []
        temp_data = []
        for feat,val in fds[i].iteritems():
            try:
                # update temp_cols iff update temp_data
                temp_cols.append(feat_dict[feat])
                temp_data.append(val)
            except KeyError as ex:
                if global_feat_dict is not None:
                    pass  # new feature in test data; nbd
                else:
                    raise ex

        # all fd's features in the same row
        k = len(temp_cols)
        cols.extend(temp_cols)
        data.extend(temp_data)
        rows.extend([i]*k)

    assert len(cols) == len(rows) and len(rows) == len(data)

    X = sparse.csr_matrix((np.array(data),
                   (np.array(rows), np.array(cols))),
                   shape=(len(fds), len(feat_dict)))
    print X
    
    # X = np.vstack([[thing] for thing in data])
    return X, feat_dict
    

## Here are two example feature-functions. They each take an xml.etree.ElementTree object, 
# (i.e., the result of parsing an xml file) and returns a dictionary mapping 
# feature-names to numeric values.
## TODO: modify these functions, and/or add new ones.
def first_last_system_call_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'first_call-x' to 1 if x was the first system call
      made, and 'last_call-y' to 1 if y was the last system call made. 
      (in other words, it returns a dictionary indicating what the first and 
      last system calls made by an executable were.)
    """
    c = Counter()
    in_all_section = False
    first = True # is this the first system call
    last_call = None # keep track of last call we've seen
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if first:
                c["first_call-"+el.tag] = 1
                first = False
            last_call = el.tag  # update last call seen
            
    # finally, mark last call seen
    c["last_call-"+last_call] = 1
    return c

def get_syscall_counts(tree):
    c = dict.fromkeys(util.dict_from_csv('systemcalls.csv'), 0)
    in_all_section = False
    
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            try: 
                c[el.tag] += 1
            except KeyError:
                continue
    return c

def system_call_count_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'num_system_calls' to the number of system_calls
      made by an executable (summed over all processes)
    """
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c['num_system_calls'] += 1
    return c

def sequence_sys_calls(tree):
    c = {'sequence': []}
    in_all_section = False
    last4 = ['', '.', ',', '!']
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section and not checkEqual(last4) and last4[0] != el.tag:
            c['sequence'] = c['sequence'] + [sys_to_int_map[el.tag]]
        last4.pop(0)
        last4.append(el.tag)
    return c['sequence']

def get_process_filesize(tree):
    c = Counter()
    for el in tree.iter():
        if el.tag == "process":
            c['filesize_all_proc'] += int(el.attrib['filesize']) * (int(not int(el.attrib['filesize']) == -1))
    return c

def get_number_timeouts(tree):
    c = Counter()
    for el in tree.iter():
        if el.tag == "process":
            c['num_terminations'] += int(el.attrib['terminationreason'] == "Timeout")
    return c

# http://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
def checkEqual(lst):
   return lst[1:] == lst[:-1]

def fraction_correct (y_hats, y):
    n = len(y)
    total = 0
    for i in xrange(n):
        total += int(y_hats[i] == y[i])
    return float(total)/float(n)

def kFoldCrossVal(k, X1, y, X2, classifier):
    kf = KFold(n_splits = k)
    bestValidation = float('inf')
    bestPred = []
    for test, train in kf.split(X1):
        X_tr, X_te, y_tr, y_te = X1[train], X1[test], y[train], y[test]
        classifier.fit(X_tr, y_tr)
        pred = classifier.predict(X_te)
        fracCorrect = fraction_correct(pred, y_te)
        if fracCorrect < bestValidation:
            bestValidation = fracCorrect
            bestPred = classifier.predict(X2)
        print "Current error: {}".format(fracCorrect)
        print "Best error: {}".format(bestValidation)
    return bestPred

def run_lstm():
    train_dir = "train"
    test_dir = "test"
    outputfile = "mypredictions.csv"  # feel free to change this or take it as an argument
    sys_to_int_map = util.dict_from_csv('systemcalls.csv')
    
    # TODO put the names of the feature functions you've defined above in this list
    ffs = [sequence_sys_calls]
    
    # extract features
    print "extracting training features..."
    X_train,t_train,train_ids = extract_lstm_feats(ffs, train_dir)
    print "done extracting training features"
    print
    
    # TODO train here, and learn your classification parameters
    print "learning..."
    # learned_W = np.random.random((len(global_feat_dict),len(util.malware_classes)))
    # create the model
    # from http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(1006, embedding_vector_length, input_length=500))
    model.add(LSTM(100))
    model.add(Dense(len(t_train[0]), activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, t_train, nb_epoch=3, batch_size=64)
    print "done learning"
    print
    
    # get rid of training data and load test data
    del X_train
    del t_train
    del train_ids
    print "extracting test features..."
    X_test,t_ignore,test_ids = extract_lstm_feats(ffs, test_dir)
    print "done extracting test features"
    print
    
    # TODO make predictions on text data and write them out
    print "making predictions..."
    # preds = np.argmax(X_test.dot(learned_W),axis=1)
    # Final evaluation of the model
    preds = model.predict(X_test, verbose=0)
    print "done making predictions"
    print
    
    print "writing predictions..."
    util.write_predictions(preds, test_ids, outputfile)
    print "done!"

def run_rf():
    train_dir = "train"
    test_dir = "test"
    outputfile = "RandomForestRegressor.csv"  # feel free to change this or take it as an argument
    sys_to_int_map = util.dict_from_csv('systemcalls.csv')
    
    # TODO put the names of the feature functions you've defined above in this list
    ffs = [get_process_filesize, get_number_timeouts, system_call_count_feats, get_syscall_counts]
    
    # extract features
    print "extracting training features..."
    X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)
    print "done extracting training features"
    print "extracting test features"
    X_test,global_feat_dict,t_test,test_ids = extract_feats(ffs, test_dir)
    print X_test
    print "done extracting test features"
    print
    
    preds = kFoldCrossVal(3, X_train.toarray(), t_train, X_test.toarray(), RandomForestClassifier())
    
    print "writing predictions..."
    util.write_predictions(preds, test_ids, outputfile)
    print "done!"

def run_svm():
    train_dir = "train"
    test_dir = "test"
    outputfile = "SVM.csv"  # feel free to change this or take it as an argument
    sys_to_int_map = util.dict_from_csv('systemcalls.csv')
    
    # TODO put the names of the feature functions you've defined above in this list
    ffs = [get_process_filesize, get_number_timeouts, system_call_count_feats, get_syscall_counts]
    
    # extract features
    print "extracting training features..."
    X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)
    print "done extracting training features"
    print "extracting test features"
    X_test,global_feat_dict,t_test,test_ids = extract_feats(ffs, test_dir)
    print X_test
    print "done extracting test features"
    print
    
    preds = kFoldCrossVal(3, X_train.toarray(), t_train, X_test.toarray(), svm.SVC())
    
    print "writing predictions..."
    util.write_predictions(preds, test_ids, outputfile)
    print "done!"

## The following function does the feature extraction, learning, and prediction
def main():
    run_svm()

if __name__ == "__main__":
    main()
