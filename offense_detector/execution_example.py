from offense_detector.Classifier import Classifier

classifier = Classifier()

classifier.train_data("data/database_racist.csv")
print(classifier.processed_data.get("tfidf_matrix"))
