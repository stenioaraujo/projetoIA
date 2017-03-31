import csv
import string

from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer


class Classifier:
    """ Classifier that puts together TfIDF and different classifiers
    """

    def __init__(self):
        self.processed_data = {
            "features": None,
            "labels": None,
            "tfidf_matrix": None,
            "split_features": None
        }
        self.gaussian_nb = GaussianNB()
        self.nb_classifier = None
        self.stemmer = SnowballStemmer("english")
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def train_data(self, raw_file_path):
        """ Load the raw_file and pre-process it.

        The process stage is:
        - stem the data
        - Remove stopwords
        - convert the remainders to a TFIDF matrix

        This will override any processed_data you have loaded.

        :param raw_file_path: the relative file path for the csv with the data.
         The csv should have two columns. It should have the header, the column
         with the data should have the name TEXT and the label ones LABEL.
        :return: self
        """
        with open(raw_file_path, "r") as open_file:
            csv_file = csv.DictReader(open_file)
            sentences_processed = []
            label = {}
            labels = []
            # (stenioaraujo) process each line and convert the labels to a
            # number. This will be used during the classification process.
            # This is gonna be used as features and labels
            for line in csv_file:
                sentences_processed.append(self.stem_sentence(line["TEXT"]))
                label[line["LABEL"]] = label.get(line["LABEL"], len(label))
                labels.append(label[line["LABEL"]])

            tfidf_matrix = self.vectorizer.fit_transform(sentences_processed)

            split_features = [s.split() for s in sentences_processed]

            self.processed_data = {
                "features": sentences_processed,
                "tfidf_matrix": tfidf_matrix,
                "labels": labels,
                "split_features": split_features
            }
        return self

    def stem_sentence(self, sentence):
        """ Stem every word in the sentence

        :param sentence: Any English sentence
        :return: the sentence with the words stemmed
        """
        sentence = sentence.translate(dict(map(lambda p: (ord(p), None),
                                               string.punctuation)))
        words_stem_list = []
        for word in sentence.split():
            words_stem_list.append(self.stemmer.stem(word))

        return " ".join(words_stem_list)
