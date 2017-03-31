import numpy as np
from time import strftime
from IPython.core.debugger import Tracer

tracer = Tracer()


def formatar_comentarios(comment):
    comment = comment.strip().strip('"')
    comment = comment.replace('_', ' ')
    comment = comment.replace('.', ' ')
    comment = comment.replace("\\\\", "\\")
    return comment


def remover_duplicadas(comments, labels):
    hashes = np.array([hash(c) for c in comments])
    unique_hashes, indices = np.unique(hashes, return_inverse=True)
    doubles = np.where(np.bincount(indices) > 1)[0]
    mask = np.ones(len(comments), dtype=np.bool)
    for i in doubles:
        not_the_first = np.where(indices == i)[0][1:]
        mask[not_the_first] = False
    return comments[mask], labels[mask]


def carregar_comentarios_treinamento(ds="train.csv"):
    print("carregando")
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
            comments.append(formatar_comentarios(comment))
    labels = np.array(labels, dtype=np.int)
    dates = np.array(dates)
    comments = np.array(comments)
    comments, labels = remover_duplicadas(comments, labels)
    return comments, labels


def carregar_dados():
    comments, labels = carregar_comentarios_treinamento("train.csv")
    comments2, labels2 = (
        carregar_comentarios_treinamento("test_with_solutions.csv"))
    comments = np.hstack([comments, comments2])
    del comments2	
    labels = np.hstack([labels, labels2])
    comments, labels = remover_duplicadas(comments, labels)
    return comments, labels


def carregar_teste(ds="test.csv"):
    print("carregando dados de treinamento")
    comments = []
    dates = []
    with open(ds) as f:
        f.readline()
        for line in f:
            splitstring = line.split(',')
            dates.append(splitstring[0][:-1])
            comment = ",".join(splitstring[1:])
            comments.append(formatar_comentarios(comment))
    comments = np.array(comments)
    return comments


def save_teste(labels, fname=None, ds="test.csv"):
    if fname is None:
        fname = "test_prediction_september_%s.csv" % strftime("%d_%H_%M")
    with open(ds) as f:
        with open(fname, 'w') as fw:
            f.readline()
            fw.write("id,Insult,Date,Commentz\n")
            for i, label, line in zip(np.arange(len(labels)), labels, f):
                fw.write("%d," % (i + 1))
                #x = 'ofensivo' if label > 0.35 else 'NãoOfensivo'
                #fw.write("%s," % x)
                fw.write("%f," % label)

                fw.write(line)