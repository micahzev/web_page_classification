"""

add hierararchical class labels to the feature sets in order to do the hierarchical classification

"""

from pymongo import MongoClient

client = MongoClient()
db = client['thesis']
fvs = db['features2']


FIRST_HIERARCHY = False  # holding_page vs non holding page

SECOND_HIERARCHY = False  # company vs non company

THIRD_HIERARCHY = False  # error vs non commercial vs other

FOURTH_HIERARCHY = False

FIFTH_HIERARCHY = False

SIXTH_HIERARCHY = False


if FIRST_HIERARCHY:
    print("FIRST_HIERARCHY")
    x = 0
    y = 0
    z = 0
    for key in fvs.find({'empty': 0}):

        if key['class_id'] == 3:
            fvs.update_one({'_id': key['_id']}, {'$set': {'hclass': 0}})
            y += 1
        else:
            fvs.update_one({'_id': key['_id']}, {'$set': {'hclass': 1}})
            z += 1
        x += 1
    print(x)
    print(y)
    print(z)
    print("COMPLETE")


if SECOND_HIERARCHY:
    print("SECOND_HIERARCHY")
    x = 0
    y = 0
    z = 0
    for key in fvs.find({'empty': 0, 'hclass': 1}):
        if key['class_id'] == 0:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h2class': 0}})
            y += 1
        elif key['class_id'] == 3:
            print("ERROR")
        else:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h2class': 1}})
            z += 1
        x += 1
    print(x)
    print(y)
    print(z)
    print("COMPLETE")


if THIRD_HIERARCHY:
    print("THIRD_HIERARCHY")
    w = 0
    x = 0
    y = 0
    z = 0
    for key in fvs.find({'empty': 0, 'h2class': 1}):
        if key['class_id'] == 1:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h3class': 0}})
            w += 1
        elif key['class_id'] == 4:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h3class': 1}})
            x += 1
        elif key['class_id'] in [3, 0]:
            print("ERROR")
        else:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h3class': 2}})
            y += 1
        z += 1
    print(w)
    print(x)
    print(y)
    print(z)
    print("COMPLETE")


if FOURTH_HIERARCHY:
    print("FOURTH_HIERARCHY")
    u = 0
    v = 0
    w = 0
    x = 0
    y = 0
    z = 0
    for key in fvs.find({'empty': 0, 'h3class': 2}):
        if key['class_id'] == 2:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h4class': 0}})
            u += 1
        elif key['class_id'] == 6:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h4class': 1}})
            v += 1
        elif key['class_id'] == 7:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h4class': 2}})
            w += 1
        elif key['class_id'] == 10:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h4class': 3}})
            x += 1
        elif key['class_id'] in [3, 0, 1, 4]:
            print("ERROR")
        else:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h4class': 4}})
            y += 1
        z += 1
    print(u)
    print(v)
    print(w)
    print(x)
    print(y)
    print(z)
    print("COMPLETE")


if FIFTH_HIERARCHY:
    print("FIFTH_HIERARCHY")
    w = 0
    x = 0
    y = 0
    z = 0
    for key in fvs.find({'empty': 0, 'h4class': 4}):
        if key['class_id'] == 5:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h5class': 0}})
            w += 1
        elif key['class_id'] == 8:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h5class': 1}})
            x += 1
        elif key['class_id'] == 9:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h5class': 1}})
            y += 1
        elif key['class_id'] in [3, 0, 1, 4, 2, 6, 7, 10]:
            print("ERROR")
        z += 1
    print(w)
    print(x)
    print(y)
    print(z)
    print("COMPLETE")


if SIXTH_HIERARCHY:
    print("SIXTH_HIERARCHY")
    x = 0
    y = 0
    z = 0
    e = 0
    for key in fvs.find({'empty': 0, 'h5class': 1}):
        if key['class_id'] == 8:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h6class': 0}})
            x += 1
        elif key['class_id'] == 9:
            fvs.update_one({'_id': key['_id']}, {'$set': {'h6class': 1}})
            y += 1
        elif key['class_id'] in [3, 0, 1, 4, 2, 6, 7, 10, 5]:
            # print("error")
            # print(key['class_id'])
            e += 1
        z += 1
    print(x)
    print(y)
    print(z)
    print(e)
    print("COMPLETE")
