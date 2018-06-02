"""
This script contains functions to know the stats of of yelp data
"""
import json
import sys


def load_bids(path):
    """

    :param path:
    :return: dictionary
    """
    bid_dict = dict()
    with open(path) as datafile:
        for line in datafile:
            business = json.loads(line)
            for category in business['categories']:
                if category not in bid_dict:
                    bid_dict[category] = set()
                bid_dict[category].add(business['business_id'])
    return bid_dict


def count_category(bpath, dpath):
    """
    Count the number of review for each category, return sorted pairs
    :param bpath: business id file path
    :param dpath: review path file
    :return:
    """
    bid_dict = load_bids(bpath)
    print("We have: "+str(len(bid_dict.keys())) +' categories.')

    results = dict.fromkeys(bid_dict.keys(), 0)
    with open(dpath) as datafile:
        for line in datafile:
            review = json.loads(line.strip())
            if review['useful'] == 0 and review['cool'] == 0 and review['funny'] == 0:
                continue
            for category in results:
                if review['business_id'] in bid_dict[category]:
                    results[category] += 1

    # sort results
    results = sorted(results.items(), key=lambda x: x[1])
    for pair in results:
        print(pair)
        # print(category[0] + ": " + str(category[1]))
    return results

count_category(sys.argv[1], sys.argv[2])