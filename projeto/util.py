import numpy as np
from time import strftime
from IPython.core.debugger import Tracer

tracer = Tracer()


def preprocess_comment(comment):
    comment = comment.strip().strip('"')
    comment = comment.replace('_', ' ')
    #comment = comment.replace('.', ' ')
    comment = comment.replace("\\\\", "\\")
    return comment


def deduplicate(comments, labels):
    hashes = np.array([hash(c) for c in comments])
    unique_hashes, indices = np.unique(hashes, return_inverse=True)
    doubles = np.where(np.bincount(indices) > 1)[0]
    mask = np.ones(len(comments), dtype=np.bool)
    # for each double entry
    for i in doubles:
        # mask out all but the first occurence
        not_the_first = np.where(indices == i)[0][1:]
        mask[not_the_first] = False
    return comments[mask], labels[mask]


def load_data(ds="train.csv"):
    print("loading")
    comments = []
    dates = []
    labels = []
    with open(ds) as f:
        f.readline()
        for line in f:
            splitstring = line.split(',')
            labels.append(splitstring[0])
            dates.append(splitstring[1][:-1])
            # the remaining commata where in the text, replace them
            comment = ",".join(splitstring[2:])
            comments.append(preprocess_comment(comment))
    labels = np.array(labels, dtype=np.int)
    dates = np.array(dates)
    comments = np.array(comments)
    comments, labels = deduplicate(comments, labels)
    return comments, labels


def load_extended_data():
    comments, labels = load_data("train.csv")
    comments2, labels2 = load_data("test_with_solutions.csv")
    comments = np.hstack([comments, comments2])
    del comments2	
    labels = np.hstack([labels, labels2])
    comments, labels = deduplicate(comments, labels)
    return comments, labels


def load_test(ds="test.csv"):
    print("loading test set")
    comments = []
    dates = []
    with open(ds) as f:
        f.readline()
        for line in f:
            splitstring = line.split(',')
            dates.append(splitstring[0][:-1])
            comment = ",".join(splitstring[1:])
            comments.append(preprocess_comment(comment))
    comments = np.array(comments)
    return comments


def write_test(labels, fname=None, ds="test.csv"):
    if fname is None:
        fname = "test_prediction_september_%s.csv" % strftime("%d_%H_%M")
    with open(ds) as f:
        with open(fname, 'w') as fw:
            f.readline()
            fw.write("id,Insult,Date,Commentz\n")
            for i, label, line in zip(np.arange(len(labels)), labels, f):
                fw.write("%d," % (i + 1))
                #x = 1 if label > 0.35 else 0
                #fw.write("XXT%dXXT," % x)
                fw.write("%f," % label)
                print(label)

                fw.write(line)