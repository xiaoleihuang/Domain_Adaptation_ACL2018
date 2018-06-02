"""
python extract_data.py Full-Economic-News-DFE-839861.csv
"""

from dateutil.parser import parse
import sys
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize


def prep_review(review_text):
    stop_set = {'when', 'who', 'being', 'again', 'these', 'his', 'some', 'or',
                'what', 'itself', 'once', 'to', 'each', 'can', 'y', 'were', 'are',
                'its', 'theirs', "you're", 'had', 'there', 'does', 'll', 'm', 'he',
                "it's", 've', 'here', 'how', 'doing', 'just', 'am', 'this',
                'while', 'having', 'more', "you'll", 'from', 'she', 'it', 'that', 'her',
                'him', 'is', 're', "she's", 'a', 'the', 'then', 's', 'such', 'those',
                'and', 'was', 'yourself', 'any', 'themselves', 'hers', 'which', "you'd", 'their',
                'an', 'at', "you've", 'where', 'd', 'them', 'for', 'will', 'same', 'o', 'be', 'have',
                'himself', 'of', 'whom', 'yourselves', 'has', 'my', 'further', 'so', 'your', 'they',
                'did', 'both', 'if', 'ain', 'ours', 'do', "that'll", 'in', 'me', 'because', 'other',
                'why', 'own', 'ourselves', 't', 'won', '\n', '\t', '-', '--'}
    return " ".join([word.strip()
                     for word in word_tokenize(review_text.strip().lower())
                     if len(word.strip()) > 0 and word not in stop_set])


def year2label(year):
    if year < 1990:
        return 1
    if year < 1995:
        return 2
    elif year < 2000:
        return 3
    elif year < 2005:
        return 4
    elif year < 2010:
        return 5
    else:
        return 6


def month2label(month):
    """
    Convert month to four season numbers
    :param month:
    :return:
    """
    if month in [1,2,3]:
        return 1
    elif month in [4,5,6]:
        return 2
    elif month in [7,8,9]:
        return 3
    else:
        return 4


datafile = pd.read_csv(sys.argv[1])
datafile = datafile[[
    'text', 'positivity', 'positivity:confidence', 'relevance',
    'relevance:confidence', 'date',
]]  # read data and extract the needed columns

rel_df = datafile[
    (datafile['relevance'].isin(['yes', 'no'])) &
    (datafile['relevance:confidence'] > 0.5)
]  # get the relevance dataset
rel_df['relevance'] = rel_df['relevance'].apply(lambda x: 1 if x == 'yes' else 0)
rel_df['year'] = rel_df['date'].apply(lambda x: parse(x).year)
# rel_df['year'] = rel_df['year'].apply(lambda x: year2label(x -100) if x > 2017 else year2label(x))
rel_df['year'] = rel_df['year'].apply(lambda x: x -100 if x > 2017 else x)
# every 8 years.
rel_df = rel_df[rel_df['year'] > 1984]
print(sorted(rel_df.year.unique()))
print(len(rel_df.year.unique()))
rel_df['year'] = rel_df['year'].apply(lambda x: year2label(x))

rel_df['month'] = rel_df['date'].apply(lambda x: month2label(parse(x).month))
rel_df['text'] = rel_df['text'].apply(lambda x: prep_review(x.replace('</br>', ' ')))

print(sorted(Counter(rel_df.year).items()))

rel_df_year = rel_df[['text', 'year', 'relevance']]
rel_df_month = rel_df[['text', 'month', 'relevance']]

# pos_df = rel_df[(rel_df['relevance'] == 1) & (rel_df['positivity:confidence'] > 0.5)]
# pos_df['positivity'] = pos_df['positivity'].apply(lambda x: 1 if x > 5 else 0)

# pos_df_year = pos_df[['text', 'year', 'positivity']]
# pos_df_month = pos_df[['text', 'month', 'positivity']]
#
# print(sorted(Counter(pos_df_year.year).items()))
# print(rel_df_year.groupby(['year', 'relevance']).size())
# print(rel_df_month.groupby(['month', 'relevance']).size())
# print(pos_df_year.groupby(['year', 'positivity']).size())
# print(pos_df_month.groupby(['month', 'positivity']).size())

new_header = ['content', 'time', 'label']
rel_df_year.columns = new_header
rel_df_month.columns = new_header
# pos_df_year.columns = new_header
# pos_df_month.columns = new_header

rel_df_year.to_csv('./economy_rel_year.tsv', sep='\t', index=False)
rel_df_month.to_csv('./economy_rel_month.tsv', sep='\t', index=False)
# pos_df_year.to_csv('./economy_pos_year.tsv', sep='\t', index=False)
# pos_df_month.to_csv('./economy_pos_month.tsv', sep='\t', index=False)