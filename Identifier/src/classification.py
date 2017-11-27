# from sklearn.multiclass import OutputCodeClassifier
# from sklearn.svm import LinearSVC
# from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

class SVMClf:
    def __init__(self, labels, data):
        self.verctorizer = DictVectorizer()
        featureVec = self.verctorizer.fit_transform(data)
        # self.classifier =  Perceptron()
        # self.classifier = OneVsRestClassifier(LinearSVC(random_state=0))
        # self.classifier = OneVsOneClassifier(LinearSVC(random_state=0))
        # self.classifier = OneVsOneClassifier(SVC(probability=True))
        # self.classifier = GaussianNB()
        self.classifier =  LogisticRegression(C=1e5)
        self.classifier.fit(featureVec, labels)
