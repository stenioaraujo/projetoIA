import numpy as np
from scipy import sparse
import re

import nltk
import nltk.collocations as col
import enchant
#from sklearn.feature_selection import SelectPercentile, chi2

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

from IPython.core.debugger import Tracer

tracer = Tracer()


def remove_non_ascii(s):
    return "".join(i for i in s if ord(i) < 128)


class DensifyTransformer(BaseEstimator):
    def fit(self, X, y=None):
        return self



class BadWordCounter(BaseEstimator):
    def __init__(self):
        with open("my_badlist.txt") as f:
            badwords = [l.strip() for l in f.readlines()]
        self.badwords_ = badwords

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        ## some handcrafted features!
        n_words = [len(c.split()) for c in documents]
        n_chars = [len(c) for c in documents]
        # number of uppercase words
        allcaps = [np.sum([w.isupper() for w in comment.split()])
               for comment in documents]
        # longest word
        max_word_len = [np.max([len(w) for w in c.split()]) for c in documents]
        # average word length
        mean_word_len = [np.mean([len(w) for w in c.split()])
                                            for c in documents]
        # number of google badwords:
        n_bad = [np.sum([c.lower().count(w) for w in self.badwords_])
                                                for c in documents]
        exclamation = [c.count("!") for c in documents]
        addressing = [c.count("@") for c in documents]
        spaces = [c.count(" ") for c in documents]

        allcaps_ratio = np.array(allcaps) / np.array(n_words, dtype=np.float)
        bad_ratio = np.array(n_bad) / np.array(n_words, dtype=np.float)

        return np.array([n_words, n_chars, allcaps, max_word_len,
            mean_word_len, exclamation, addressing, spaces, bad_ratio, n_bad,
            allcaps_ratio]).T


class FeatureStacker(BaseEstimator):
    """Stacks several transformer objects to yield concatenated features.
    Similar to pipeline, a list of tuples ``(name, estimator)`` is passed
    to the constructor.
    """
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def get_feature_names(self):
        pass

    def fit(self, X, y=None):
        for name, trans in self.transformer_list:
            trans.fit(X, y)
        return self

    def transform(self, X):
        features = []
        for name, trans in self.transformer_list:
            features.append(trans.transform(X))
        issparse = [sparse.issparse(f) for f in features]
        if np.any(issparse):
            features = sparse.hstack(features).tocsr()
        else:
            features = np.hstack(features)
        return features

