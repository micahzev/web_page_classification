"""

simple analysis investigating how and why so many empty sites are coming through the scraper

"""

import csv

from pymongo import MongoClient

client = MongoClient()
db = client['thesis']
raws = db['raws']

# domainscan files are lists of labeled urls

with open('domainscan.csv', 'r') as f:
    reader = csv.reader(f)
    data1 = list(reader)


with open('domainscan_2016.csv', 'r') as f:
    reader = csv.reader(f)
    data2 = list(reader)

data = data1+data2

empty = []
non_empty = []

nodata = []

x = 0

for webpage in data[1:]:

    site = "http://" + webpage[0] + "." + webpage[1] + "/"

    if raws.find({'site': site}).count() == 1:

        res = raws.find_one({'site':site})

        if res['raw'] == '<html><head></head><body></body></html>':
            empty.append(webpage[2])
        else:
            non_empty.append(webpage[2])

    elif raws.find({'site': site}).count() > 1:

        print("MULTIPLE")
        print(site)

    elif raws.find({'site': site}).count() == 0:

        nodata.append(site)

    else:

        print("Error: ")
        print(site)

    x += 1


print(x)

print("NODATA")
print(nodata)

print("EMPTIES:")
print(len(empty))
empty_freq = {i: empty.count(i) for i in set(empty)}
print(empty_freq)


print("NONEMPTIES:")
print(len(non_empty))
non_empty_freq = {i: non_empty.count(i) for i in set(non_empty)}
print(non_empty_freq)
