"""

there are number of empty sites scraped by the scraper for a variety of reasons

this script will reattempt to scrape any sites that came up empty

it will also try different domain name substrings

"""

import time
import platform

from selenium import webdriver
from pymongo import MongoClient

start_time = time.time()

# os specific phantom file
if platform.system() == 'Windows':
    PHANTOMJS_PATH = './phantomjs.exe'
else:
    PHANTOMJS_PATH = r'./phantomjs'

# connect to MongoDB collection
client = MongoClient()
db = client['thesis']
raws = db['raws']

# raw scraped data for empty sites will look like this

empty_site_string = '<html><head></head><body></body></html>'

N = raws.find({'raw': empty_site_string}).count()

print("Number of empty sites: " + str(N))

for idx, item in enumerate(raws.find({'raw': empty_site_string})):

    cont = 0

    elapse = time.time() - start_time

    print("opening: " + item['site'] + " || elapsed time: " + str(elapse) + " seconds " + str(idx+1) + " of " + str(N))

    browser = webdriver.PhantomJS(PHANTOMJS_PATH)
    browser.set_page_load_timeout(600)
    try:

        browser.get(item['site'])

        raws.update_one({'site': item['site']}, {'$set': {'raw': browser.page_source}})

        cont = 1

    except:

        print("FAILED TO UPDATE: " + item['site'])

    if cont == 0:
        browser.quit()
        continue

    else:

        # if not able to find site without www, then try with the www substring

        browser = webdriver.PhantomJS(PHANTOMJS_PATH)

        # timeout is 10 minutes
        browser.set_page_load_timeout(600)

        splitname = item['site'].split('//')

        www_sitename = '//www.'.join(splitname)

        try:

            browser.get(www_sitename)

            raws.update_one({'site': item['site']}, {'$set': {'raw': browser.page_source}})

        except:

            print("FAILED TO UPDATE: " + item['site'])

        browser.quit()
