"""

scrapes data from a list of labeled URLs read in from a csv

sites are first rendered with PhantomJS

scraped rendered data is stored in MongoDB

"""

import csv
import time
import platform

from selenium import webdriver
from pymongo import MongoClient

START_TIME = time.time()


# os specific phantom file
if platform.system() == 'Windows':
    PHANTOMJS_PATH = './phantomjs.exe'
else:
    PHANTOMJS_PATH = r'./phantomjs'

########################################################
#
# 2015-2016 set of labeled data
#
# csv has the form:
# <SITE DOMAIN NAME>,<TOP LEVEL DOMAIN NAME>,<CATEGORY>
#
# example:
# google,com,company
#
########################################################


INPUT_FILE = 'domainscan_2016.csv'

with open(INPUT_FILE, 'r') as f:
    reader = csv.reader(f)
    labeled_url_data = list(reader)


# connect to MongoDB collection

client = MongoClient()
db = client['thesis']
raws = db['raws']

for webpage in labeled_url_data[1:]:

    site = "http://" + webpage[0] + "." + webpage[1] + "/"

    # has this site been added already?
    if raws.find({'site': site}).count() == 0:

        browser = webdriver.PhantomJS(PHANTOMJS_PATH)

        # timeout is 10 minutes
        browser.set_page_load_timeout(60)

        try:

            browser.get(site)

            raws.insert_one({'site': site, 'raw': browser.page_source, 'set': 2016})

        except:

            # timeout
            print("FAILED TO OPEN (timeout): " + site)

        browser.quit()

    elif raws.find({'site': site}).count() == 1:

        # already in the db
        continue

    else:
        # error due to to MongoDB
        print("Mongo error: " + site)

print("Runtime: " + str(time.time() - START_TIME) + " seconds")
