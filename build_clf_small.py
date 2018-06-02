from utils import data_helper
import pickle
from random_domain import lr_domain_random
import os

file_list = [
    # './data/vaccine/vaccine_month_sample.tsv',
    # './data/vaccine/vaccine_year_sample.tsv',
    # './data/parties/parties_year_sample.tsv',
    # './data/aware/aware_month_sample.tsv',
    # './data/economy/economy_rel_month_sample.tsv',
    # './data/economy/economy_rel_year_sample.tsv',
    './data/amazon/amazon_review_month_sample.tsv',
    './data/amazon/amazon_review_year_sample.tsv',
    './data/yelp/yelp_Hotels_month_sample.tsv',
    './data/yelp/yelp_Hotels_year_sample.tsv',
    './data/yelp/yelp_Restaurants_month_sample.tsv',
    './data/yelp/yelp_Restaurants_year_sample.tsv',
]

for data_path in file_list:
    dataset = data_helper.load_data(data_path)
    paths = data_path.split('/')
    paths[1] = 'features'
    paths[-1] = paths[-1][:-11]
    outp = '/'.join(paths)
    for ftype in ['tfidf', 'binary']:
        tmp_path = outp+'_'+ftype+'.pkl'
        tmp_path = data_helper.train_fvs_da(dataset, outputfile=outp, balance=False, fea_type=ftype)
        #fvs_file = pickle.load(open(tmp_path, 'rb'))

        print(tmp_path)
        #for balance in [True, False]:
        #    print('\t----------------Balance: ' + str(balance) + '---------------')
        #    lr_domain_random(fvs_file, balance=balance)
        #print()
        print()
