"""

all features are constructed from raw scraped html

"""

import csv
import re
import regex
from time import time

from pymongo import MongoClient

from bs4 import BeautifulSoup, Comment

from urllib.parse import urlparse

# natural language toolkit
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import snowball

# custom language detection algorithm
import lang_detect as ld

# connect to MongoDB collection
# raws are the raw scraped data (INPUT)
# fvs are the construced feature vectors (OUTPUT)

client = MongoClient()
db = client['thesis']
raws = db['raws']
fvs = db['features2']

t0 = time()

# get all names from the input data set

with open('domainscan.csv', 'r') as f:
    reader = csv.reader(f)
    data1 = list(reader)

with open('domainscan_2016.csv', 'r') as f:
    reader = csv.reader(f)
    data2 = list(reader)

data = data1 + data2

# toggle debug mode (will stop saving to db collection)
debug = True

for webpage in data[1:]:

    # fv is the main feature vector object we are building build

    fv = {}

    # class label of the site

    klass = webpage[2]

    site = "http://" + webpage[0] + "." + webpage[1] + "/"

    source = raws.find_one({'site': site})

    if not source:
        print("Could not locate: " + site)
        continue

    # BeautifulSoup parses the raw site html

    soup = BeautifulSoup(source['raw'], "lxml")

    # build the feature vector with extracted information

    if source['raw'] == '<html><head></head><body></body></html>':\

        # am empty site feature vector

        fv['empty'] = 1

        fv['site'] = site

        fv['class'] = klass

        fv['set'] = source['set']

        if not debug:
            fvs.insert_one(fv)

        continue

    else:

        # not empty

        fv['empty'] = 0

    fv['site'] = site

    fv['class'] = klass

    fv['set'] = source['set']

    # full page text
    soup_text = soup

    # extract features from <script> tags

    scripts = soup_text.findAll('script')

    fv['number_of_scripts'] = len(scripts)

    script_sources = [link['src'] for link in soup_text.findAll('script', src=True)]

    # extract features from <meta> tags

    metas = soup_text.findAll('meta')

    meta_types = [tag.attrs['name'] for tag in metas if 'name' in tag.attrs]

    meta_types = [item.lower() for item in meta_types]

    fv['l_metas'] = len(metas)

    if 'viewport' in meta_types:
        fv['meta_viewport'] = 1
    else:
        fv['meta_viewport'] = 0

    if 'description' in meta_types:
        fv['meta_description'] = 1
    else:
        fv['meta_description'] = 0

    if 'robot' in meta_types or 'robots' in meta_types:
        fv['meta_robot'] = 1
    else:
        fv['meta_robot'] = 0

    if 'author' in meta_types:
        fv['meta_author'] = 1
    else:
        fv['meta_author'] = 0

    if 'language' in meta_types:
        fv['meta_language'] = 1
    else:
        fv['meta_language'] = 0

    if 'title' in meta_types:
        fv['meta_title'] = 1
    else:
        fv['meta_title'] = 0

    if 'generator' in meta_types:
        fv['meta_generator'] = 1
    else:
        fv['meta_generator'] = 0

    meta_text = ' '.join([tag.attrs['content'] for tag in metas
                          if 'name' in tag.attrs and
                          tag.attrs['name'] in ['title', 'description'] and
                          'content' in tag.attrs])

    fv['meta_text'] = meta_text

    fv['number_of_scripts_with_source'] = len(script_sources)

    # extract features from <style> tags

    styles = [x for x in soup_text.findAll('style')]

    fv['number_of_styles'] = len(styles)

    # extract features from <link> tags

    links = [x for x in soup_text.findAll('link')]

    fv['number_of_links'] = len(links)

    link_href = [link['href'] for link in soup_text.findAll('link') if 'href' in link.attrs]

    parsed_links = [urlparse(item) for item in link_href]

    link_file_types = [item.path.split(".")[-1] for item in parsed_links
                       if "." in item.path and
                       '/' not in item.path.split(".")[-1]]

    link_file_types_freq = {i: link_file_types.count(i) for i in set(link_file_types)}

    fv['link_file_types_freq'] = link_file_types_freq

    stylez = [item for item in link_href if 'css' in item]
    otherz = [item for item in link_href if 'css' not in item]

    fv['no_style_css_link_tags'] = len(stylez)
    fv['no_non_css_link_tags'] = len(otherz)

    # remove <script>, <noscript> and <style> tags going forward for cleaner parsing

    [x.extract() for x in soup_text.findAll('script')]
    [x.extract() for x in soup_text.findAll('noscript')]
    [x.extract() for x in soup_text.findAll('style')]

    # extract features from <ul> tags

    lists = soup_text.findAll('ul')

    fv['number_of_lists'] = len(lists)

    # extract features from comments <!--- --->

    commentz = soup_text.find_all(string=lambda txt: isinstance(txt, Comment))

    fv['no_of_comments'] = len(commentz)

    all_comments_length = [len(comment) for comment in commentz]

    fv['average_comment_length'] = sum(all_comments_length)/len(commentz) if len(commentz) != 0 else 0

    # constructing features from text data

    # all site text
    text = soup_text.get_text()

    # add meta_text
    text = text + ' ' + meta_text

    fv['text'] = ''.join(text.split())

    # lowercase text
    fv['text_lc'] = fv['text'].lower()

    fv['text_length'] = len(fv['text'])

    # tokenize text

    fv['tokenized_text'] = wordpunct_tokenize(fv['text_lc'])

    fv['number_of_tokens'] = len(fv['tokenized_text'])

    # detect language

    fv['language'] = ld.detect_language(fv['text_lc'])

    # remove stop words

    # collect words into different filters: punctuation, stopwords, non_stopword (aka normal text), and numbers

    fv['punctuation'] = []
    fv['stopwords'] = []
    fv['non_stopwords'] = []
    fv['numericals'] = []

    for item in fv['tokenized_text']:

        # check if number
        number = re.match(r'^[0-9]+$', item)

        # check if alphabetical

        alpha = regex.match(r'^\p{L}+$', item)

        if item in stopwords.words(fv['language']):
            fv['stopwords'].append(item)
        elif number:
            fv['numericals'].append(item)
        elif alpha:
            fv['non_stopwords'].append(item)
        else:
            fv['punctuation'].append(item)

    punctuation_freq = {i: fv['punctuation'].count(i) for i in set(fv['punctuation'])}

    keys_with_dot = [item for item in punctuation_freq.keys() if '.' in item or '$' in item]

    for key in keys_with_dot:
        intermediate_key = key.replace('.', 'P')
        new_key = intermediate_key.replace('$', 'D')
        punctuation_freq[new_key] = punctuation_freq.pop(key)

    fv['punctuation_freq'] = punctuation_freq

    fv['numerical_freq'] = {i: fv['numericals'].count(i) for i in set(fv['numericals'])}

    fv['stopword_freq'] = {i: fv['stopwords'].count(i) for i in set(fv['stopwords'])}

    # stemming

    fv['stemmed'] = []

    # set default stemmer

    stemmer = snowball.EnglishStemmer()

    if fv['language'] == 'english':
        stemmer = snowball.EnglishStemmer()
    elif fv['language'] == 'french':
        stemmer = snowball.FrenchStemmer()
    elif fv['language'] == 'dutch':
        stemmer = snowball.DutchStemmer()
    elif fv['language'] == 'hungarian':
        stemmer = snowball.HungarianStemmer()
    elif fv['language'] == 'portuguese':
        stemmer = snowball.PortugueseStemmer()
    elif fv['language'] == 'danish':
        stemmer = snowball.DanishStemmer()
    elif fv['language'] == 'german':
        stemmer = snowball.GermanStemmer()

    for word in fv['non_stopwords']:
        if stemmer:
            fv['stemmed'].append(stemmer.stem(word))
        else:
            fv['stemmed'].append(word)

    # site domain is another feature

    domain_name = webpage[1]

    fv['dom'] = domain_name

    # all html tag data (names and frequencies)

    tag_data = [tag.name for tag in soup.find_all()]

    all_tags = list(set(tag_data))

    fv['all_tags'] = all_tags

    tag_freq = {i: tag_data.count(i) for i in set(tag_data)}

    fv['tag_freq'] = tag_freq

    # extract features from <a> tags
    # there are a lot of features extracted from links

    anchor_tags = [link.get('href') for link in soup.find_all('a')]

    # text from anchor tags
    anchor_text = [link.text for link in soup.find_all('a')]

    # images in anchor text
    anchor_images = [link.find('img') for link in soup.find_all('a') if link.find('img') is not None]

    fv['anchor_tags'] = anchor_tags

    fv['anchor_text'] = ' '.join(anchor_text)

    fv['l_anchor_images'] = len(anchor_images)

    # empty tages, either have None, # / or ''

    none_tags = [item for item in anchor_tags if item is None]

    fv['l_none_tags'] = len(none_tags)

    empty_string_tags = [item for item in anchor_tags if item == '']

    fv['l_empty_string_tags'] = len(empty_string_tags)

    empty_hash_tags = [item for item in anchor_tags if item == "#"]

    fv['l_empty_hash_tags'] = len(empty_hash_tags)

    root_nav = [item for item in anchor_tags if item == "/"]

    fv['l_root_nav'] = len(root_nav)

    temp_excludes = none_tags + empty_hash_tags + root_nav + empty_string_tags

    # parsed urls from href in <a> tag

    parted = [urlparse(item) for item in anchor_tags if item not in temp_excludes]

    http = [item for item in parted if item.scheme == 'http']

    fv['l_http'] = len(http)

    https = [item for item in parted if item.scheme == 'https']

    fv['l_https'] = len(https)

    mailto = [item for item in parted if item.scheme == 'mailto']

    fv['l_mailto'] = len(mailto)

    tel = [item for item in parted if item.scheme == 'tel']

    fv['l_tel'] = len(tel)

    javascript = [item for item in parted if item.scheme == 'javascript']

    fv['l_javascript'] = len(javascript)

    maps = [item for item in parted if item.netloc == 'maps.google.com']

    fv['l_maps'] = len(maps)

    # social media related links

    facebook = [item for item in parted if 'facebook' in item.netloc]

    fv['l_facebook'] = len(facebook)

    twitter = [item for item in parted if 'twitter' in item.netloc]

    fv['l_twitter'] = len(twitter)

    linkedin = [item for item in parted if 'linkedin' in item.netloc]

    fv['l_linkedin'] = len(linkedin)

    youtube = [item for item in parted if 'youtube' in item.netloc]

    fv['l_youtube'] = len(youtube)

    pinterest = [item for item in parted if 'pinterest' in item.netloc]

    fv['l_pinterest'] = len(pinterest)

    instagram = [item for item in parted if 'instagram' in item.netloc]

    fv['l_instagram'] = len(instagram)

    google = [item for item in parted if 'google' in item.netloc]

    fv['l_google'] = len(google)

    google_plus = [item for item in parted if 'plus.google' in item.netloc]

    fv['l_google_plus'] = len(google_plus)

    googleplay = [item for item in parted if 'play.google' in item.netloc]

    fv['l_google_play'] = len(googleplay)

    # navigation <a> tags

    navigate = [item for item in parted if item.fragment != "" and item.scheme == "" and
                item.netloc == "" and item.path == "" and item.params == ""]

    fv['l_navigate'] = len(navigate)

    # links to files

    file_extensions = [item.path.split(".")[-1] for item in parted if "." in item.path]

    fv['file_ext_freq'] = {i: file_extensions.count(i) for i in set(file_extensions)}

    # domains linked to

    domain_extensions = [item.netloc.split(".")[-1] for item in parted if item.netloc.split(".")[-1]]

    fv['dom_ext_freq'] = {i: domain_extensions.count(i) for i in set(domain_extensions)}

    # relative links vs absolute links

    relative_link = [item for item in parted if item.path != "" and item.scheme == "" and item.netloc == ""]

    fv['l_relative_link'] = len(relative_link)

    temp_excludes = mailto + javascript + navigate + tel

    sub_level = [item for item in parted if item not in temp_excludes]

    snippet_domain = webpage[0]

    domain = webpage[0] + "." + webpage[1]

    # internal links vs external links (domain-wise)

    internal_links = relative_link + [item for item in sub_level if domain in item.netloc]

    fv['l_internal_links'] = len(internal_links)

    related_internal_links = relative_link + [item for item in sub_level if snippet_domain in item.netloc]

    fv['l_related_internal_links'] = len(related_internal_links)

    external_links = [item for item in sub_level if item not in relative_link and item not in internal_links]

    fv['l_external_links'] = len(external_links)

    strictly_external_links = [item for item in external_links if item not in related_internal_links]

    fv['l_strictly_external_links'] = len(strictly_external_links)

    absolute_link = [item for item in internal_links if item.scheme in ['http', 'https']]

    fv['l_absolute_link'] = len(absolute_link)

    # indexes built from looking at social media links and business related links

    fv['social_media_score'] = fv['l_facebook'] + fv['l_twitter'] + fv['l_linkedin'] + \
                               fv['l_pinterest'] + fv['l_instagram'] + fv['l_google_plus']

    fv['business_score'] = fv['l_mailto'] + fv['l_tel'] + fv['l_maps'] + fv['l_linkedin']

    # extract features from tags with 'src' attribute (also considered linked)

    sources = [item['src'] for item in soup.findAll(src=True)]

    source_tag_types = [item.name for item in soup.findAll(src=True)]

    fv['sources'] = sources

    parsed_sources = [urlparse(item) for item in sources]

    parsed_source_file_types = [item.path.split(".")[-1] for item in parsed_sources
                                if "." in item.path and
                                '/' not in item.path.split(".")[-1]]

    parsed_source_file_types_freq = {i: parsed_source_file_types.count(i) for i in set(parsed_source_file_types)}

    fv['src_file_type_freq'] = parsed_source_file_types_freq

    fv['source_tag_types'] = source_tag_types

    fv['src_tag_types_freq'] = {i: source_tag_types.count(i) for i in set(source_tag_types)}

    # end of feature construction, add to db
    if not debug:
        fvs.insert_one(fv)

print(time()-t0)
print("finished")
