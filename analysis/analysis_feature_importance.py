"""

analysis of feature importances from Random Forest

gives insight into which features are important

"""


import csv
import operator
from pprint import pprint as p

from pymongo import MongoClient

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


init = {
 'anchor_text': [],
 'meta_text': [],
 'average_comment_length': [],
 'business_score': [],
 'dom_ext_freq': [],
 'file_ext_freq': [],
 'l_absolute_link': [],
 'l_anchor_images': [],
 'l_empty_hash_tags': [],
 'l_empty_string_tags': [],
 'l_external_links': [],
 'l_facebook': [],
 'l_google': [],
 'l_google_play': [],
 'l_google_plus': [],
 'l_http': [],
 'l_https': [],
 'l_instagram': [],
 'l_internal_links': [],
 'l_javascript': [],
 'l_linkedin': [],
 'l_mailto': [],
 'l_maps': [],
 'l_metas': [],
 'l_navigate': [],
 'l_none_tags': [],
 'l_pinterest': [],
 'l_related_internal_links': [],
 'l_relative_link': [],
 'l_root_nav': [],
 'l_strictly_external_links': [],
 'l_tel': [],
 'l_twitter': [],
 'l_youtube': [],
 'link_file_types_freq': [],
 'meta_author': [],
 'meta_description': [],
 'meta_generator': [],
 'meta_language': [],
 'meta_robot': [],
 'meta_title': [],
 'meta_viewport': [],
 'no_non_css_link_tags': [],
 'no_of_comments': [],
 'no_style_css_link_tags': [],
 'number_of_links': [],
 'number_of_lists': [],
 'number_of_scripts': [],
 'number_of_scripts_with_source': [],
 'number_of_styles': [],
 'number_of_tokens': [],
 'numerical_freq': [],
 'punctuation_freq': [],
 'set': [],
 'social_media_score': [],
 'src_file_type_freq': [],
 'src_tag_types_freq': [],
 'stemmed': [],
 'stopword_freq': [],
 'tag_freq': [],
 'text_length': []
}


agg = {
 'anchor_text': [],
 'meta_text': [],
 'average_comment_length': [],
 'business_score': [],
 'dom_ext_freq': [],
 'file_ext_freq': [],
 'l_absolute_link': [],
 'l_anchor_images': [],
 'l_empty_hash_tags': [],
 'l_empty_string_tags': [],
 'l_external_links': [],
 'l_facebook': [],
 'l_google': [],
 'l_google_play': [],
 'l_google_plus': [],
 'l_http': [],
 'l_https': [],
 'l_instagram': [],
 'l_internal_links': [],
 'l_javascript': [],
 'l_linkedin': [],
 'l_mailto': [],
 'l_maps': [],
 'l_metas': [],
 'l_navigate': [],
 'l_none_tags': [],
 'l_pinterest': [],
 'l_related_internal_links': [],
 'l_relative_link': [],
 'l_root_nav': [],
 'l_strictly_external_links': [],
 'l_tel': [],
 'l_twitter': [],
 'l_youtube': [],
 'link_file_types_freq': [],
 'meta_author': [],
 'meta_description': [],
 'meta_generator': [],
 'meta_language': [],
 'meta_robot': [],
 'meta_title': [],
 'meta_viewport': [],
 'no_non_css_link_tags': [],
 'no_of_comments': [],
 'no_style_css_link_tags': [],
 'number_of_links': [],
 'number_of_lists': [],
 'number_of_scripts': [],
 'number_of_scripts_with_source': [],
 'number_of_styles': [],
 'number_of_tokens': [],
 'numerical_freq': [],
 'punctuation_freq': [],
 'set': [],
 'social_media_score': [],
 'src_file_type_freq': [],
 'src_tag_types_freq': [],
 'stemmed': [],
 'stopword_freq': [],
 'tag_freq': [],
 'text_length': []
}

for item in feats5000.find():
    for key in item:
        if item[key] != [] and key != '_id':
            for elem in item[key]:
                init[key].append(elem)
            # # init[key].append(item[key][0])
            #
            # init[key].append(item[key])


for key in init:

    aggregate = {}

    for item in init[key]:

        if item[1] in aggregate:

            aggregate[item[1]][0] += float(item[0])
            aggregate[item[1]][1] += 1

        else:

            aggregate[item[1]] = [float(item[0]), 1]

    agg[key] = aggregate

for key in agg:

    for item in agg[key]:
        agg[key][item] = agg[key][item][0] / float(agg[key][item][1])

long_list = []

for key in agg:
    for item in agg[key]:
        long_list.append([key, item, agg[key][item]])


sorted_long = sorted(long_list, key=lambda x: x[2], reverse=True)


importances = [item[2] for item in sorted_long]


with open("feature_importance.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(sorted_long[:250])


def freq_ouput(input_list):

    return {i: input_list.count(i) for i in set(input_list)}

for key in init:
    init[key] = freq_ouput(init[key])


aggregate = {}

for key in init:
    if init[key] != {}:
        for innerkey in init[key]:
            if innerkey == key:
                aggregate[key+", "] = init[key][innerkey]
            else:
                aggregate[key+", "+innerkey] = init[key][innerkey]


sorted_out = sorted(aggregate.items(), key=operator.itemgetter(1), reverse=True)

p(sorted_out)
