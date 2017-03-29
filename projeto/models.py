from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from features import BadWordCounter, FeatureStacker
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


def build_elasticnet_model():
    select = SelectPercentile(score_func=chi2, percentile=16)

    clf = SGDClassifier(loss='log', penalty="elasticnet", shuffle=True,
            alpha=0.0001, l1_ratio=0.95, n_iter=20)
    countvect_char = TfidfVectorizer(ngram_range=(1, 5),
            analyzer="char", binary=False)
    countvect_word = TfidfVectorizer(ngram_range=(1, 3),
            analyzer="word", binary=False, min_df=3)
    badwords = BadWordCounter()
    scaler = MinMaxScaler()
    badwords_pipe = Pipeline([('bad', badwords), ('scaler', scaler)])

    ft = FeatureStacker([("badwords", badwords_pipe), ("chars",
        countvect_char), ("words", countvect_word)])
    pipeline = Pipeline([('vect', ft), ('select', select), ('logr', clf)])
    return pipeline

