"""

experimental  hierarchical classifier using random forest in multiclass setting with feature selection

very similar to the hierarchical_learn.py script but with feature selection using feature importance

"""

text_input = (
    'stemmed',
    'anchor_text',
    'meta_text'
)

freq_input = (
    'dom_ext_freq',
    'file_ext_freq',
    'link_file_types_freq',
    'punctuation_freq',
    'numerical_freq',
    'src_file_type_freq',
    'src_tag_types_freq',
    'stopword_freq',
    'tag_freq',
)

categorical_input = (
    'language',
)

numerical_input = (
    'average_comment_length',
    'business_score',
    'l_absolute_link',
    'l_anchor_images',
    'l_empty_hash_tags',
    'l_empty_string_tags',
    'l_external_links',
    'l_facebook',
    'l_google',
    'l_google_play',
    'l_google_plus',
    'l_http',
    'l_https',
    'l_instagram',
    'l_internal_links',
    'l_javascript',
    'l_linkedin',
    'l_mailto',
    'l_maps',
    'l_metas',
    'l_navigate',
    'l_none_tags',
    'l_pinterest',
    'l_related_internal_links',
    'l_relative_link',
    'l_root_nav',
    'l_strictly_external_links',
    'l_tel',
    'l_twitter',
    'l_youtube',
    'meta_author',
    'meta_description',
    'meta_generator',
    'meta_language',
    'meta_robot',
    'meta_title',
    'meta_viewport',
    'no_non_css_link_tags',
    'no_of_comments',
    'no_style_css_link_tags',
    'number_of_links',
    'number_of_lists',
    'number_of_scripts',
    'number_of_scripts_with_source',
    'number_of_styles',
    'number_of_tokens',
    'set',
    'social_media_score',
    'text_length',
)


# these will change if using hierarchical learning
class_label = 'class_id'
# class_label = 'hclass'
# class_label = 'h2class'
# class_label = 'h3class'
# class_label = 'h4class'
# class_label = 'h5class'
# class_label = 'h6class'

unusued_features = (
    '_id',
    'all_tags',
    'anchor_tags',
    'class',
    'dom',
    'empty',
    'non_stopwords',
    'numericals',
    'punctuation',
    'site',
    'source_tag_types',
    'sources',
    'stopwords',
    'text',
    'text_lc',
    'tokenized_text',
)
