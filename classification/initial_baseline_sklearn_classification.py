"""

initial baseline using a series of different classification algorithms:

Extra Trees Classifier
k-Nearest Neighbours Classifier
Random Forest Classifier
Support Vector Machine
Voting
Guassian Naive Bayes
Multinomial Naive Bayes

"""

from __future__ import print_function

from time import time

import numpy as np

import matplotlib.pyplot as plt

from pymongo import MongoClient

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

# main features table of all features
from main_features import text_input, freq_input, numerical_input, class_label

# connect to db

client = MongoClient()
db = client['thesis']
fvs = db['features2']

initial_sets = []
labels = []

for item in fvs.find({'empty': 0}):

    item_add = {}

    for feature in text_input + freq_input:
        item_add[feature] = item[feature]

    for feature in numerical_input:
        item_add[feature] = np.array([item[feature]])

    initial_sets.append(item_add)

    labels.append(item[class_label])


# custom sklearn selector class
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

# train test split is 0.66 / 0.33
X_train, X_test, y_train, y_test = train_test_split(initial_sets, labels, test_size=0.33, random_state=42)

text_union = []
freq_union = []
numerical_union = []

for feature in text_input:
    text_union.append(
        (feature,
            Pipeline(
                [
                    ('selector', ItemSelector(key=feature)),
                    ('tfidf',  TfidfVectorizer(sublinear_tf=True, min_df=7, max_df=.2,  ngram_range=(1, 3))),
                ]
            )
         )
    )

for feature in freq_input:
    freq_union.append(
        (feature,
             Pipeline(
                 [
                    ('selector', ItemSelector(key=feature)),
                    ('vect', DictVectorizer(sparse=True)),
                ]
             )
         )
    )

for feature in numerical_input:
    numerical_union.append(
        (feature,
         Pipeline(
             [
                 ('selector', ItemSelector(key=feature))
             ]
         )
        )
    )


unionized = text_union + freq_union + numerical_union

union = FeatureUnion(transformer_list=unionized)


X_train_flip = {}
X_test_flip = {}

for feature in text_input+freq_input:

    X_train_flip[feature] = [item[feature] for item in X_train]

    X_test_flip[feature] = [item[feature] for item in X_test]


for feature in numerical_input:

    X_train_flip[feature] = np.array([item[feature] for item in X_train])

    X_test_flip[feature] = np.array([item[feature] for item in X_test])


target_names = [
    'company',
    'error',
    'for sale',
    'holding page',
    'non-commercial',
    'password protected',
    'pay-per-click',
    'personal-family-blog',
    'porn',
    'portal/media',
    'web-shop'
    ]


# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf[1])
    t0 = time()

    pipeline = Pipeline([

        ('union', union),

        # ("MinMaxScaler", MaxAbsScaler()),

        # ("StandardScaler", StandardScaler(with_mean=False)),

        # ("normalise", Normalizer()),

        clf

    ])

    pipeline.fit(X_train_flip, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = pipeline.predict(X_test_flip)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(pipeline).split('(')[0]

    return clf_descr, score, train_time, test_time

extra_tree = ('Extratree', ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0))
kNN = ("kNN", KNeighborsClassifier(n_neighbors=10))
random_forest = ("Random forest", RandomForestClassifier(n_estimators=100))
svc_l1 = ("Linear SCV L1", LinearSVC(loss='l2', penalty='l1', dual=False, tol=1e-3))
scv_l2 = ("Linear SCV L2", LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3))
feature_reduction_svc_l1 = ("feature_reduction_svc_l1", Pipeline([
  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
  ('classification', LinearSVC())
]))
featur_reduction_svc_l2 = ("featur_reduction_svc_l2", Pipeline([
  ('feature_selection', LinearSVC(penalty="l2", dual=False, tol=1e-3)),
  ('classification', LinearSVC())
]))

soft_voter = ("soft_voter",
              VotingClassifier(estimators=[extra_tree, kNN, random_forest], voting='soft', weights=[2, 1, 2]))
hard_voter = ("hard_voter",
              VotingClassifier(estimators=[extra_tree, kNN, random_forest, svc_l1, scv_l2], voting='hard'))

results = []

param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


results.append(benchmark(random_forest))


gauss = ("Guass", GaussianNB(priors=None))

multi = ("multi", MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None))

results.append(benchmark(gauss))

kNN = ("kNN", KNeighborsClassifier(n_neighbors=10, metric=cosine_similarity))


results.append(benchmark(kNN))

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
