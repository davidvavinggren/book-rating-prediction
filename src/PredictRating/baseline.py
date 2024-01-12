from PredictRating.constants import *
from PredictRating.classes.data import Data

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report as report
from sklearn.metrics import confusion_matrix as conf_mat
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main(model):
    data = Data('reviews.json', p = DATA_PERCENTAGE, split = TRAIN_SPLIT)

    clf = make_pipeline(CountVectorizer(), model)
    clf.fit(X = data.train['review'], y = data.train['rating'])

    correct = data.test['rating']
    preds = clf.predict(data.test['review'])
    
    cmat = conf_mat(correct, preds)
    acc = cmat.diagonal().sum() / cmat.sum()
    print(f'acc = {acc}')
    print(report(y_true = correct, y_pred = preds))
    disp = ConfusionMatrixDisplay(cmat, display_labels = ['1', '2', '3', '4', '5'])
    disp.plot()
    plt.show()

if __name__ == '__main__':
    model_mnb = MultinomialNB()
    model_svc = LinearSVC()

    main(model_svc)