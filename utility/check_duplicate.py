"""

simply checks for duplicate labeled sites in the input data

"""

import csv

with open('domainscan_2016.csv', 'r') as f:
    reader = csv.reader(f)
    data17 = list(reader)

with open('domainscan.csv', 'r') as f:
    reader = csv.reader(f)
    data16 = list(reader)


d16 = [x[0] for x in data16[1:]]
d17 = [x[0] for x in data17[1:]]


dupl = []

for item in d16:
    if item in d17:
        dupl.append(item)


for item in d17:
    if item in d16:
        dupl.append(item)

dupl = set(dupl)

dupl = list(dupl)

print(dupl)

print("2016")

for item in data16:
    if item[0] in dupl:
        print(item)

print("")

print("2017")

for item in data17:
    if item[0] in dupl:
        print(item)
