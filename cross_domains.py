"""
This script is to hold out one entire domain as test data and train on the other all domains.
"""

import pandas as pd
import pickle
import sys
from utils import model_helper
from sklearn.metrics.classification import f1_score
from scipy.sparse import vstack

large_pkl = pickle.loads(open(sys.argv[1], 'rb'))


# loop through each big dataset
for data_name in large_pkl:
    print('Working on data: ' + data_name)

    # test each feature type
    for ftype in ['binary', 'tfidf']:
        print('Testing on feature type: ' + ftype)

        # record performance in each domain
        record_base = dict()
        record_da = dict()
        # test through each domain
        for domain in large_pkl[data_name]['uniq_domains']:
            """test base"""
            clf = model_helper.build_lr_clf()
            # load data
            dataset = pd.read_csv(large_pkl[data_name]['raw'], sep='\t')
            # formalize train_data and test_data
            test_data = pickle.load(open(large_pkl[data_name]['da_vect'][domain][ftype], 'rb'))
            test_label = pd.read_csv(large_pkl[data_name][domain]).label


            for tmp_d in large_pkl[data_name]['uniq_domains']:
                if tmp_d == domain:
                    continue

                train_data = pd.read_csv(large_pkl[data_name]['data'][tmp_d], sep='\t')
                train_label = dataset[dataset.time != domain].label


            # convert to features

            # fit classifier

            # test
