"""
python extract_data.py /home/public_data/yelp/dataset/business.json /home/public_data/yelp/dataset/review.json
"""

import json
import sys
from nltk.tokenize import word_tokenize
from dateutil.parser import parse
from nltk import ngrams
from langdetect import detect

category = 'Hotels' # "Hotels", "Restaurants"
bids = set() # buisiness ids.

# first load business ids.
with open(sys.argv[1]) as datafile:
    for line in datafile:
        business = json.loads(line)
        if category in business['categories']:
            bids.add(business['business_id'])

month_file = open('yelp_'+category+'_month.tsv', 'w')
month_file.write('content\ttime\tlabel\n')
year_file = open('yelp_'+category+'_year.tsv', 'w')
year_file.write('content\ttime\tlabel\n')


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
                'why', 'own', 'ourselves', 't', 'won', '\n', '\t'}
    return " ".join([word.strip()
                     for word in word_tokenize(review_text.strip().lower())
                     if len(word.strip()) > 0 and word not in stop_set])


def year2label(year):
    if year < 2009:
        return 1
    elif year < 2012:
        return 2
    elif year < 2015:
        return 3
    else:
        return 4


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


def create_ngram(sent, n=3):
    for idx in range(2, n+1):
        ng = ngrams(sent.split(), idx)
        sent = sent + " " + ' '.join([term for term in ['_'.join(tmp_pair) for tmp_pair in ng]])
    return sent


dsets = set()
# then load the review dataset and filtered by the ids
with open(sys.argv[2]) as datafile:
    for line in datafile:
        review = json.loads(line.strip())
        if review['business_id'] not in bids:
            continue
#        if review['useful'] == 0 and review['cool'] == 0 and review['funny'] == 0:
#            continue
#        if review['stars'] == 3:
#            continue


        content = review['text'].replace('\n', '')
        content = content.replace('\t', '')
        content = prep_review(content)
        if len(content.split()) < 10:
            continue
#        if detect(content) != 'en': # filter out non-English languages
#            continue


        date = parse(review['date'])
        dsets.add(date.year)
        if review['stars'] > 3:
            label = '1'
        else:
            label = '0'

        #content = create_ngram(content)
        year_file.write(content +'\t'+str(year2label(date.year))+'\t'+label+'\n')
        month_file.write(content + '\t' + str(month2label(date.month)) + '\t' + label + '\n')

print(sorted(list(dsets)))

month_file.close()
year_file.close()
