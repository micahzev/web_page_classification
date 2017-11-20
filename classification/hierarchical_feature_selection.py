"""

experimental  hierarchical classifier using random forest in multiclass setting with feature selection

very similar to the hierarchical_learn.py script but with feature selection using feature importance

"""

from __future__ import print_function

import itertools
from time import time

import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from main_features import text_input, freq_input, numerical_input, class_label

from pymongo import MongoClient

# collect all our feature data
client = MongoClient()
db = client['thesis']
fvs = db['features2']
feats10 = db['feats10']
feats50 = db['feats50']
feats100 = db['feats100']
feats500 = db['feats500']
feats1000 = db['feats1000']
feats2000 = db['feats2000']

reduction = db['selected']

feature_selection = reduction.find_one()

initial_sets = []
labels = []


# custom sklearn selector class
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


# custom sklearn selector class
class ReducedItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self):
        return self

    def transform(self, data_dict):
        # this fancy dictionary comprehension will filter out non selected features from our feature dictionaries!
        return [{k: item[k] for k in item.keys() & feature_selection[self.key]} for item in data_dict[self.key]]


# building hierarchies
# variable hierarchies is the structured multilabel classifier protocol

target_names_0 = [
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

target_names_1 = [
    'holding_page',
    'non_holding_page'
]

target_names_2 = [
    'company',
    'non_company'
]

target_names_3 = [
    'error',
    'non_commmercial',
    'other',
]


target_names_4 = [
    'for sale',
    'pay-per-click',
    'personal-family-blog',
    'web-shop',
    'other'
]


target_names_5 = [
    'password protected',
    'other'
]

target_names_6 = [
    'porn',
    'portal/media',
]


# there are a series of different strategies that were experimented with
# define the hierarchical strategy in the hiearchies list

# hierarchies = [
#
#     # [ 'full', 'class_id', {'empty':0},              0.33, target_names_0 ],
#
#     # [ 'H1',   'hclass',   {'empty':0},              0.33, target_names_1 ],
#     #
#     # [ 'H2',   'h2class',  {'empty':0, 'hclass':1},  0.33, target_names_2 ],
#     #
#     # [ 'H3',   'h3class',  {'empty':0, 'h2class':1}, 0.33, target_names_3 ],
#
#     [ 'H4',   'h4class',  {'empty':0, 'h3class':2}, 0.33,  target_names_4 ],
#
#     [ 'H5',   'h5class',  {'empty':0, 'h4class':4}, 0.33,  target_names_5 ],
#
#     [ 'H6',   'h6class',  {'empty':0, 'h5class':1}, 0.33,  target_names_6 ],
#
# ]
#
# hierarchies = [
#
#     [ 'english', 'class_id', {'empty':0, 'language':'english'},              0.33, target_names_0 ],
#
#     ['dutch', 'class_id', {'empty': 0, 'language': 'dutch'}, 0.33, target_names_0],
#
#     ['french', 'class_id', {'empty': 0, 'language': 'french'}, 0.33, target_names_0],
#
#     ['german', 'class_id', {'empty': 0, 'language': 'german'}, 0.33, target_names_0],
#
#     ['other', 'class_id',
#      {'empty': 0,
#       'language': {
#             '$nin' : ['english','dutch','french']
#         }
#       },
#      0.33, target_names_0],
#
#
# ]

hierarchies = [

    # [ 'full', 'class_id', {'empty':0},              0.33, target_names_0 ],
    #
    # [ 'H1',   'hclass',   {'empty':0},              0.33, target_names_1 ],
    #
    # [ 'H2',   'h2class',  {'empty':0, 'hclass':1},  0.33, target_names_2 ],
    #
    # [ 'H3',   'h3class',  {'empty':0, 'h2class':1}, 0.33, target_names_3 ],
    #
    # [ 'H4',   'h4class',  {'empty':0, 'h3class':2}, 0.2,  target_names_4 ],
    #
    # [ 'H5',   'h5class',  {'empty':0, 'h4class':4}, 0.2,  target_names_5 ],

    ['H6',   'h6class',  {'empty': 0, 'h5class': 1}, 0.2,  target_names_6],

]

# hierarchies = [
#
#     ['H4', 'h4class', {'empty': 0, 'h3class': 2}, 0.2, target_names_4],
#
#     ['H5', 'h5class', {'empty': 0, 'h4class': 4}, 0.2, target_names_5],
#
#     ['H6', 'h6class', {'empty': 0, 'h5class': 1}, 0.2, target_names_6],
#
#     [ 'H4_33',   'h4class',  {'empty':0, 'h3class':2}, 0.33,  target_names_4 ],
#
#     [ 'H5_33',   'h5class',  {'empty':0, 'h4class':4}, 0.33,  target_names_5 ],
#
#     [ 'H6_33',   'h6class',  {'empty':0, 'h5class':1}, 0.33,  target_names_6 ],
#
#
# ]

for settings in hierarchies:

    initial_sets = []
    labels = []

    print("##########################################")
    print("HIERARCHY: " + settings[0])
    print("##########################################")

    for item in fvs.find(settings[2]):

        item_add = {}

        for feature in text_input + freq_input:
            item_add[feature] = item[feature]

        for feature in feature_selection['hand_builts']:
            item_add[feature] = np.array([item[feature]])

        initial_sets.append(item_add)

        labels.append(item[settings[1]])

    # X_train, X_test, y_train, y_test = train_test_split(initial_sets, labels, test_size=0.33, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(initial_sets, labels, test_size=settings[3], random_state=42)

    print('')

    text_union = []
    freq_union = []
    numerical_union = []

    for feature in text_input:
        text_union.append(
            (feature,
             Pipeline(
                 [
                     ('selector', ItemSelector(key=feature)),
                     ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=7, max_df=.2, ngram_range=(1, 3),
                                               vocabulary=feature_selection[feature])),
                 ]
             )
             )
        )

    for feature in freq_input:
        freq_union.append(
            (feature,
             Pipeline(
                 [
                     ('selector', ReducedItemSelector(key=feature)),
                     ('vect', DictVectorizer(sparse=True)),
                 ]
             )
             )
        )

    for feature in feature_selection['hand_builts']:
        numerical_union.append(
            (feature,
             Pipeline(
                 [
                     ('selector', ItemSelector(key=feature))
                 ]
             )
             )
        )

    unionized = text_union+freq_union+numerical_union

    union = FeatureUnion(transformer_list=unionized)

    X_train_flip = {}
    X_test_flip = {}

    for feature in text_input+freq_input:

        X_train_flip[feature] = [item[feature] for item in X_train]

        X_test_flip[feature] = [item[feature] for item in X_test]

    for feature in feature_selection['hand_builts']:
        X_train_flip[feature] = np.array([item[feature] for item in X_train])

        X_test_flip[feature] = np.array([item[feature] for item in X_test])

    pipeline = Pipeline([

        ('union', union),

        ("Random forest", RandomForestClassifier(n_estimators=2000, n_jobs=-1)),

    ])

    print('')


    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        np.set_printoptions(precision=2)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.around(cm, 2)
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max()
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def benchmark(clf):
        print('_' * 80)
        print(clf.steps[1][0])
        print('_' * 80)
        print("Training: ")
        # print(clf)
        t0 = time()
        clf.fit(X_train_flip, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(X_test_flip)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)

        # f1score = metrics.f1_score(y_test, pred,average='micro')
        # print("f1-score:   %0.3f" % f1score)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))

            print()

        print("classification report:")
        print(metrics.classification_report(y_test, pred, target_names=settings[4]))

        # print("confusion matrix:")
        cm = metrics.confusion_matrix(y_test, pred)
        # print(cm)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=settings[4],
                              title='Confusion matrix')

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=settings[4], normalize=True,
                              title='Normalized confusion matrix')

        plt.show()

        # print()
        # print("roc auc score:")
        # roc_auc_score = metrics.roc_auc_score(y_test, pred, average='micro')
        # print(roc_auc_score)

        # print("roc curve:")
        # roc_curve = metrics.roc_curve(y_test, pred)
        # print(roc_curve)

        print()
        clf_descr = str(clf).split('(')[0]
        return clf, clf_descr, score, train_time, test_time


    benchmark(pipeline)
