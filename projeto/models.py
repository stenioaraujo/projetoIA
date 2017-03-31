from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from features import BadWord, FeaturePilha
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


def criar_classificador():
    selecao = SelectPercentile(score_func=chi2, percentile=16)

    classificador = SGDClassifier(loss='log', penalty="elasticnet",
                                  shuffle=True, alpha=0.0001, l1_ratio=0.95,
                                  n_iter=20)
    tfidf_char = TfidfVectorizer(ngram_range=(1, 5), analyzer="char",
                                 binary=False)
    tfidf_palavras = TfidfVectorizer(ngram_range=(1, 3), analyzer="word",
                                     binary=False, min_df=3)
    badwords = BadWord()
    scaler = MinMaxScaler()
    badwords_pipe = Pipeline([('bad', badwords), ('scaler', scaler)])
    fp = FeaturePilha([("badwords", badwords_pipe), ("chars",
                                                     tfidf_char), ("words", tfidf_palavras)])
    pipeline = Pipeline([('vect', fp), ('select', selecao), ('logr', classificador)])
    return pipeline
