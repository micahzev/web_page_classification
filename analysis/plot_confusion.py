"""

plotting confusion matrices of results of classifier

"""

import itertools

import numpy as np

import matplotlib.pyplot as plt

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
    'holding page',
    'non holding page'
]

target_names_2 = [
    'company',
    'non company'
]

target_names_3 = [
    'error',
    'non commercial',
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

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

targets = target_names_5

# cm = np.array([
#     [0.66,0.03,0.17,0.0,0.14],
#     [0.06,0.85,0.1,0.0,0.0],
#     [0.01,0.02,0.94,0.0,0.02],
#     [0.03,0.0,0.33,0.6,0.03],
#     [0.04,0.0,0.57,0.06,0.34]
#
# ])

conf_m = np.array([
    [0.6, 0.4],
    [0.28, 0.72]
])

plt.figure()
plot_confusion_matrix(conf_m, classes=targets, normalize=True)

plt.show()
