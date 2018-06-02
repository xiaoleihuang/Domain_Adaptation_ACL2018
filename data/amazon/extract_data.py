"""This script is to extract Amazon Review dataset"""
import json

from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk import ngrams
from langdetect import detect


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
                'why', 'own', 'ourselves', 't', 'won', '``'}
    return " ".join([word.strip() for word in word_tokenize(review_text.strip().lower()) if len(word.strip())>0 and word not in stop_set])


def create_ngram(sent, n=3):
    for idx in range(2, n+1):
        ng = ngrams(sent.split(), idx)
        sent = sent + " " + ' '.join([term for term in ['_'.join(tmp_pair) for tmp_pair in ng]])
    return sent


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


def year2label(year):
    # the data was splitted into three time periods:
    # [(1997, 1999), (2000, 2002), (2003, 2005), (2006, 2008),
    # (2009, 2011), (2012, 2014)]
    if year < 2000:
        return 1
    elif year < 2003:
        return 2
    elif year < 2006:
        return 3
    elif year < 2009:
        return 4
    elif year < 2012:
        return 5
    else:
        return 6


def amazon2csv(data_path):
    """"""
    date_min = 1000000000000
    date_max = 0
    with open('amazon_review_year.tsv', 'w') as year_file:
        with open('amazon_review_month.tsv', 'w') as month_file:
            year_file.write('\t'.join(['content', 'time', 'label']) + '\n')
            month_file.write('\t'.join(['content', 'time', 'label']) + '\n')

            with open(data_path) as datafile:
                for line in datafile:
                    one_review = json.loads(line.strip())

                    date_tmp = datetime.utcfromtimestamp(one_review['unixReviewTime'])
                    comment_tmp = one_review['reviewText'].strip()
                    score_tmp = one_review['overall']
                    comment_tmp = prep_review(comment_tmp).strip()

                    # filter
#                    if score_tmp == 3: # remove neutral ones
#                        continue
                    if len(comment_tmp.split()) < 10: # at least 10 tokens
                        continue
                    if one_review['helpful'][0] < 1: # helpfulness
                        continue
                    if detect(comment_tmp) != 'en': # english language
                        continue

                    if score_tmp <= 3:
                        score_tmp = '0'
                    else:
                        score_tmp = '1'
                    
                    
                    if one_review['unixReviewTime'] < date_min:
                        date_min = one_review['unixReviewTime']
                    if one_review['unixReviewTime'] > date_max:
                        date_max = one_review['unixReviewTime']

                    year_file.write('\t'.join([comment_tmp, str(year2label(date_tmp.year)), score_tmp])+'\n')
                    month_file.write('\t'.join([comment_tmp, str(month2label(date_tmp.month)), score_tmp])+'\n')

                print(datetime.utcfromtimestamp(date_min))
                print(datetime.utcfromtimestamp(date_max))

if __name__ == '__main__':
    """python extract_data.py /home/public_data/amazon_reviews/CDs_and_Vinyl_5.json"""
    import sys
    amazon2csv(sys.argv[1])
