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


def remover_caracteres_estranhos(s):
    return "".join(i for i in s if ord(i) < 128)


class Densidade(BaseEstimator):
    def fit(self, X, y=None):
        return self


class BadWord(BaseEstimator):
    def __init__(self):
        with open("my_badlist.txt") as f:
            badwords = [l.strip() for l in f.readlines()]
        self.badwords_ = badwords

    def fit(self, comentarios, y=None):
        return self

    def transform(self, comentarios):
        numero_palavras = [len(c.split()) for c in comentarios]
        numero_caracteres_comentario = [len(c) for c in comentarios]
        # numero de palavras tudo maiusculo
        maiusculas = [np.sum([w.isupper() for w in comentario.split()])
               for comentario in comentarios]
        # palavra com maior comprimento
        maior_comprimento_palavras = [np.max([len(w) for w in c.split()])
                                      for c in comentarios]
        # tamanho medio das palavras
        tamanho_medio = [np.mean([len(w) for w in c.split()])
                                            for c in comentarios]
        # numero de plavras que batem com as palavras ruim do google
        google_palavras = [np.sum([c.lower().count(w) for w in self.badwords_])
                           for c in comentarios]
        num_exclamacao = [c.count("!") for c in comentarios]
        referencia_alguem = [c.count("@") for c in comentarios]
        espacoes = [c.count(" ") for c in comentarios]

        media_tudo_maiusculo = (
            np.array(maiusculas) / np.array(numero_palavras, dtype=np.float))

        palavras_google_media = (
            np.array(google_palavras) / np.array(numero_palavras,dtype=np.float))

        return np.array([numero_palavras, numero_caracteres_comentario,
                         maiusculas, maior_comprimento_palavras, tamanho_medio,
                         num_exclamacao, referencia_alguem, espacoes,
                         palavras_google_media, google_palavras,
                         media_tudo_maiusculo]).T


class FeaturePilha(BaseEstimator):
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def get_feature_names(self):
        pass

    def fit(self, X, y=None):
        for name, trans in self.transformer_list:
            trans.fit(X, y)
        return self

    def transform(self, X):
        caracteristicas = []
        for name, trans in self.transformer_list:
            caracteristicas.append(trans.transform(X))
        issparse = [sparse.issparse(f) for f in caracteristicas]
        if np.any(issparse):
            caracteristicas = sparse.hstack(caracteristicas).tocsr()
        else:
            caracteristicas = np.hstack(caracteristicas)
        return caracteristicas

