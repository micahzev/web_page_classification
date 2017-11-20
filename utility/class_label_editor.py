"""

utility script to make specific changes to the mongodb dataset

"""

from pymongo import MongoClient

client = MongoClient()
db = client['thesis']
raws = db['raws']
fvs = db['features2']
cl = db['class_labels']

# set the action you want here

BUILD_HASH_TABLE = False

MERGE_HOLDING_PAGE_LABELS = False

SET_TRAIN_TEST = False

SET_CLASS_LABELS_TO_HASH_IDENTITIES = False

SET_DATA_SOURCE = False

JOIN_STEMMED = True


# build a hash table for reference of categories, convert classes into numbers

if BUILD_HASH_TABLE:

    class_dict = {'company': 0,
                  'error': 1,
                  'for sale': 2,
                  'holding page': 3,
                  'non-commercial': 4,
                  'password protected': 5,
                  'pay-per-click': 6,
                  'personal-family-blog': 7,
                  'porn': 8,
                  'portal/media': 9,
                  'web-shop': 10}

    cl.insert_one(class_dict)

# decided to merge two similar classes into one superclass

if MERGE_HOLDING_PAGE_LABELS:
    print("MERGE_HOLDING_PAGE_LABELS")
    x = 0
    for key in fvs.find():
        if key['class'] == 'holding page non-commercial':
            fvs.update_one({'_id': key['_id']}, {'$set': {'class': 'holding page'}})
            x += 1
        elif key['class'] == 'holding page company':
            fvs.update_one({'_id': key['_id']}, {'$set': {'class': 'holding page'}})
            x += 1

    print(x)
    print("COMPLETE")


# construct custom train test split (although I recommend using sklearn's function

if SET_TRAIN_TEST:
    print("SET_TRAIN_TEST")
    for key in fvs.find({'empty': 0}):
        sorter = random.randint(1, 100)
        site_name = key['site']
        if sorter < 76:
            fvs.update_one({'site': site_name}, {'$set': {'train': 1}})
        else:
            fvs.update_one({'site': site_name}, {'$set': {'train': 0}})
    print("COMPLETE")


# use the hash table to set the class labels to class_id's

if SET_CLASS_LABELS_TO_HASH_IDENTITIES:
    print("SET_CLASS_LABELS_TO_HASH_IDENTITIES")
    x = 0
    for key in fvs.find():
        ID = cl.find_one()[key['class']]
        entry_id = key['_id']
        fvs.update_one({'_id': entry_id}, {'$set': {'class_id': ID}})
        x += 1
    print(x)
    print("COMPLETE")

# set the data source of the datasets either to 2016 or 2017

if SET_DATA_SOURCE:
    for key in raws.find():
        site_name = key['site']
        raws.update_one({'site': site_name}, {'$set': {'set': 2016}})

# stemmed text was tokenize and to make it easier to work with the list of tokens can be joined into one long sentence

if JOIN_STEMMED:
    print("JOIN_STEMMED")
    x = 0
    for key in fvs.find({'empty': 0}):

        joined_text = ' '.join(key['stemmed'])

        fvs.update_one({'_id': key['_id']}, {'$set': {'stemmed': joined_text}})
        x += 1
    print(x)
    print("COMPLETE")
