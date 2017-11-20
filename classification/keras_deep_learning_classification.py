"""

classification of input feature vectors using Keras Deep Learning algorithms

"""

import numpy as np

from pymongo import MongoClient

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

# main features table of all features
from main_features import text_input, freq_input, numerical_input, class_label


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# connect to db

client = MongoClient()
db = client['thesis']
fvs = db['features2']

initial_sets = []
labels = []

# 'empty' 0 are non-empty sites
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


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)


kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


results = cross_val_score(estimator, X_train, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
