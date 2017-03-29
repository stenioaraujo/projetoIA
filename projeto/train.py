import numpy as np
import matplotlib as mpl
mpl.use('Agg')

from models import build_elasticnet_model


from util import load_extended_data, write_test, load_test

from IPython.core.debugger import Tracer


tracer = Tracer()



def apply_models():
    comments, labels = load_extended_data()
    comments_test = load_test("testes_para_saida.csv")

    clf2 = build_elasticnet_model()
    probs_common = np.zeros((len(comments_test), 2))
    clf2.fit(comments, labels)
    probs = clf2.predict_proba(comments_test)
    # print("score: %f" % auc_score(labels_test, probs[:, 1]))
    probs_common += probs
    write_test(probs[:, 1], "saida.csv",
               ds="testes_para_saida.csv")



if __name__ == "__main__":
    #explore_features()
    apply_models()
