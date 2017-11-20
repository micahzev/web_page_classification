"""

base Random Forest Learning algorithm from which the hierarchical learn and feature selection algorithms are copied

"""

from __future__ import print_function

from time import time
import itertools
from pprint import pprint as p
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

from pymongo import MongoClient

from main_features import text_input, freq_input, numerical_input, class_label


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
feats5000 = db['feats5000']

reduction = db['selected']

SHOULD_I_REDUCE = False

if SHOULD_I_REDUCE:
    # clean the collection
    reduction.drop()


# custom sklearn selector class
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


search_requirements = {
    'empty': 0  # i only want non empty sites
    # 'language':'french'
    # 'set':2017,
}

initial_sets = []
labels = []


for item in fvs.find(search_requirements):

    item_add = {}

    for feature in text_input + freq_input:
        item_add[feature] = item[feature]

    for feature in numerical_input:
        item_add[feature] = np.array([item[feature]])

    initial_sets.append(item_add)

    labels.append(item[class_label])


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


unionized = text_union+freq_union+numerical_union

union = FeatureUnion(transformer_list=unionized)

X_train_flip = {}
X_test_flip = {}

for feature in text_input+freq_input:

    X_train_flip[feature] = [item[feature] for item in X_train]

    X_test_flip[feature] = [item[feature] for item in X_test]


for feature in numerical_input:

    X_train_flip[feature] = np.array([item[feature] for item in X_train])

    X_test_flip[feature] = np.array([item[feature] for item in X_test])

# labels
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

pipeline = Pipeline([

    ('union', union),

    ("Random forest", RandomForestClassifier(n_estimators=2000, n_jobs=-1, oob_score=True)),
    # ("Random forest", RandomForestClassifier(n_estimators=1000, n_jobs=-1, oob_score=True,class_weight='balanced')),


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
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
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
    print(clf)
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
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    cm = metrics.confusion_matrix(y_test, pred)
    print(cm)

    print()
    clf_descr = str(clf).split('(')[0]
    return clf, clf_descr, score, train_time, test_time


benchmark(pipeline)

feature_groups = []
feature_length = []
feature_names = []
feature_set = []

for item in union.transformer_list:

    feature_groups.append(item[0])

    try:
        feature_length.append(len(item[1]._final_estimator.vocabulary_))
        feature_names.append(item[1]._final_estimator.get_feature_names())
        feature_set.append([item[0]]*len(item[1]._final_estimator.vocabulary_))
    except AttributeError:
        feature_length.append(1)
        feature_names.append([item[1]._final_estimator.key])
        feature_set.append([item[0]])

feature_cumsum = np.cumsum(feature_length)

flattened_feature_names = [item for sublist in feature_names for item in sublist]

flattened_feature_set = [item for sublist in feature_set for item in sublist]


zipped = zip(feature_groups, feature_length, feature_cumsum)

verts = []

for item in list(zipped):
    p(item)
    # if item[2] < 14598:
    verts.append(item[2])

print(verts)

fi = pipeline.named_steps['Random forest'].feature_importances_


normed = fi/np.linalg.norm(fi)

sortz = list(reversed(np.sort(normed)))


feature_analysis = np.array(list(zip(normed, fi, flattened_feature_names, flattened_feature_set)))

sorted_feature_analysis = sorted(feature_analysis, key=lambda x: float(x[0]))


###########################
#
#   FEATURE SELECTION CODE
#
###########################

# choose K highest features

K = -1000


selected_features = sorted_feature_analysis[K:]

selection = {}

selection['stemmed'] = []
selection['anchor_text'] = []
selection['meta_text'] = []

selection['dom_ext_freq'] = []
selection['file_ext_freq'] = []
selection['link_file_types_freq'] = []
selection['punctuation_freq'] = []
selection['numerical_freq'] = []
selection['src_file_type_freq'] = []
selection['src_tag_types_freq'] = []
selection['stopword_freq'] = []
selection['tag_freq'] = []

selection['hand_builts'] = []

for item in selected_features:
    if item[3] == 'stemmed':
        selection['stemmed'].append(item[2])
    elif item[3] == 'anchor_text':
        selection['anchor_text'].append(item[2])
    elif item[3] == 'meta_text':
        selection['meta_text'].append(item[2])
    elif item[3] == 'dom_ext_freq':
        selection['dom_ext_freq'].append(item[2])
    elif item[3] == 'file_ext_freq':
        selection['file_ext_freq'].append(item[2])
    elif item[3] == 'link_file_types_freq':
        selection['link_file_types_freq'].append(item[2])
    elif item[3] == 'punctuation_freq':
        selection['punctuation_freq'].append(item[2])
    elif item[3] == 'numerical_freq':
        selection['numerical_freq'].append(item[2])
    elif item[3] == 'src_file_type_freq':
        selection['src_file_type_freq'].append(item[2])
    elif item[3] == 'src_tag_types_freq':
        selection['src_tag_types_freq'].append(item[2])
    elif item[3] == 'stopword_freq':
        selection['stopword_freq'].append(item[2])
    elif item[3] == 'tag_freq':
        selection['tag_freq'].append(item[2])
    else:
        selection['hand_builts'].append(item[2])


if SHOULD_I_REDUCE:
    reduction.insert_one(selection)

# build feature analysis

list_of_settings = [[-10, feats10], [-50, feats50], [-100, feats100],
                    [-500, feats500], [-1000, feats1000], [-2000, feats2000]]

# list_of_settings = [[-5000,feats5000]]

for setting in list_of_settings:

    feature_analyst = {}

    for group in feature_groups:
        feature_analyst[group] = []

    for item in sorted_feature_analysis[setting[0]:]:
        feature_analyst[item[3]].append([item[0], item[2]])

    setting[1].insert_one(feature_analyst)

bin_analysis = [[key, len(feature_analyst[key])] for key in feature_analyst]

p(bin_analysis)
