import os
import sys
from utils import data_helper, model_helper
import argparse


if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument('--fea_type', default='tfidf', type=str, help='Type of features: binary or tfidf')
     parser.add_argument('--balance', default=False, type=bool)
     args = parser.parse_args()
     print(args)
     print()

    file_list = [
        ('./data/amazon/amazon_month_sample.tsv', './features/amazon/amazon_review_month_tfidf.pkl', {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 10}),
        ('./data/economy/economy_month_sample.tsv', './features/economy/economy_rel_month_tfidf.pkl', {'C': 1, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 10}),
        ('./data/vaccine/vaccine_month_sample.tsv', './features/vaccine/vaccine_month_tfidf.pkl', {'C': 1, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 10}),
        ('./data/yelp/yelp_Hotels_month_sample.tsv', './features/yelp/yelp_Hotels_month_tfidf.pkl', {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 10}),
        ('./data/yelp/yelp_Restaurants_month_sample.tsv', './features/yelp/yelp_Restaurants_month_tfidf.pkl', {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 10}),
         ('./data/amazon/amazon_year_sample.tsv', './features/amazon/amazon_review_year_tfidf.pkl', {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 30}),
         ('./data/economy/economy_year_sample.tsv', './features/economy/economy_rel_year_tfidf.pkl', {'C': 1, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 100}),
         ('./data/parties/parties_year_sample.tsv', './features/parties/parties_year_tfidf.pkl', {'C': 1, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 100}),
         ('./data/vaccine/vaccine_year_sample.tsv', './features/vaccine/vaccine_year_tfidf.pkl', {'C': 1, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 100}),
         ('./data/yelp/yelp_Hotels_year_sample.tsv', './features/yelp/yelp_Hotels_year_tfidf.pkl', {'C': 1, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 30}),
         ('./data/yelp/yelp_Restaurants_year_sample.tsv', './features/yelp/yelp_Restaurants_year_tfidf.pkl', {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 100}),
    ]

    # loop through each pair
    for pair in file_list:
        print('Working on: ' + pair[0])
        if not os.path.exists(pair[1]):
            print('feature file does not exist, train a feature vector')
            data_path = pair[0]
            dataset = data_helper.load_data(data_path)

            outp = data_helper.train_fvs_da(dataset, outputfile=pair[1], balance=args.balance, fea_type=args.fea_type)
            print('New feature pickle file has been saved to: '+ str(outp))
