"""
Test on one domain, and train on the other domains,
Output f1 scores and visualize them by heat map
"""
from utils import data_helper, model_helper

from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

import numpy as np
np.random.seed(0)
from pandas import DataFrame

import seaborn as sns
import matplotlib.pyplot as plt
import argparse


def domain2year(domain, name):
    domain = int(domain)
    if 'vaccine' == name:
        if domain == 2013:
            return '2013'
        elif domain == 2014:
            return '2014'
        elif domain == 2015:
            return '2015'
        elif domain == 2016:
            return '2016'
    elif 'amazon' in name:
        if domain == 1:
            return '1997-99'
        elif domain == 2:
            return '2000-02'
        elif domain == 3:
            return '2003-05'
        elif domain == 4:
            return '2006-08'
        elif domain == 5:
            return '2009-11'
        elif domain == 6:
            return '2012-14'
    elif 'yelp' in name:
        if domain == 1:
            return '2006-08'
        elif domain == 2:
            return '2009-11'
        elif domain == 3:
            return  '2012-14'
        elif domain == 4:
            return '2015-17'
    elif 'economy' in name:
        if domain == 1:
            return '1985-89'
        elif domain == 2:
            return '1990-94'
        elif domain == 3:
            return '1995-99'
        elif domain == 4:
            return '2000-04'
        elif domain == 5:
            return '2005-09'
        elif domain == 6:
            return '2010-14'
    elif 'parties' in name:
        if domain == 1:
            return '1948-56'
        elif domain == 2:
            return '1960-68'
        elif domain == 3:
            return '1972-80'
        elif domain == 4:
            return '1984-92'
        elif domain == 5:
            return '1996-2004'
        elif domain == 6:
            return '2008-16'


def domain2month(domain, name=None):
    if domain == 1:
        return 'Jan-Mar'
    elif domain == 2:
        return 'Apr-Jun'
    elif domain == 3:
        return 'Jul-Sep'
    else:
        return 'Oct-Dec'


def cross_test_domain_clf(dataset, domain2label, data_name=None, balance=False, binary=False, ):
    """
    Train on one domain, test on others
    :return:
    """
    uniq_domains = sorted(list(set([item[-2] for item in dataset])))
    results = DataFrame([[0.0]*len(uniq_domains)]*len(uniq_domains),
                        index=[domain2label(item, data_name) for item in uniq_domains],
                        columns=[domain2label(item, data_name) for item in uniq_domains])
    print(uniq_domains)

    # loop through each domain
    for domain in uniq_domains:

        # build train_data
        train_x = []
        train_y = []
        for item in dataset:
            if domain == item[-2]:
                train_x.append(item[0])
                train_y.append(item[-1])

        # build vectorizer and encoder
        label_encoder = LabelEncoder()
        if len(dataset) > 15469: # this number is length of "./yelp/yelp_Hotels_year_sample.tsv" - 1000
            if not binary:
                vectorizer = TfidfVectorizer(min_df=2, tokenizer=lambda x: x.split())
            else:
                vectorizer = TfidfVectorizer(min_df=2, tokenizer=lambda x: x.split(),
                                             binary=True, use_idf=False, smooth_idf=False)
        else:
            if not binary:
                vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2)
            else:
                vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 3),
                                             binary=True, use_idf=False, smooth_idf=False)

        # encode the data
        train_y = label_encoder.fit_transform(train_y)
        train_x = vectorizer.fit_transform(train_x)

        # balance
        if balance:
            random_sampler = RandomOverSampler(random_state=0)
            train_x, train_y = random_sampler.fit_sample(train_x, train_y)

        # build classifier
        clf = model_helper.build_lr_clf()
        clf.fit(train_x, train_y)

        # instead of skipping self-domain, we take the 5-fold cross-validation for this domain
        results[domain2label(domain, data_name)][domain2label(domain, data_name)] = np.mean(
            cross_val_score(model_helper.build_lr_clf(),
                            train_x, train_y, cv=5,
                            scoring='f1_weighted')
        )
        train_x = None
        train_y = None


        # test and evaluation
        for test_domain in [item for item in uniq_domains if item != domain]:
            if int(test_domain) == int(domain):
                continue

            test_x = []
            test_y = []
            for item in dataset:
                if test_domain == item[-2]:
                    test_x.append(item[0])
                    test_y.append(item[-1])

            # encode the data
            test_y = label_encoder.transform(test_y)
            test_x = vectorizer.transform(test_x)
            tmp_result = str(f1_score(y_true=test_y, y_pred=clf.predict(test_x), average='weighted'))
            # results[domain][test_domain] = str(f1_score(y_true=test_y, y_pred=clf.predict(test_x), average='weighted'))
            # print(str(domain)+','+str(test_domain)+','+str(f1_score(y_true=test_y, y_pred=clf.predict(test_x), average='weighted')))
            results[domain2label(test_domain, data_name)][domain2label(domain, data_name)] = tmp_result

            test_x = None
            test_y = None

    # pickle.dump(results, open('cross_test_domain_results_'+str(balance)+'.pkl', 'wb'))
    print(results)
    return results


def viz_perform(df, title, outpath='./image/output.pdf'):
    """
    Heatmap visualization
    :param df: an instance of pandas DataFrame
    :return:
    """
    a4_dims = (11.7, 11.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.set(font_scale=1.2)
    viz_plot = sns.heatmap(df, annot=True, cbar=False, ax=ax, annot_kws={"size": 24}, cmap="YlGnBu", vmin=df.values.min(), fmt='.3f')
    plt.xticks(rotation=20, fontsize=25)
    plt.xlabel('Train', fontsize=25)
    plt.ylabel('Test', fontsize=25)
    plt.title(title, fontsize=25)
    viz_plot.get_figure().savefig(outpath, format='pdf')
    plt.close()


if __name__ == '__main__':
    """
    
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--month', default=None,
    #                     type=str, help='The path raw csv or tsv file')
    # parser.add_argument('--year', default=None,
    #                     type=str, help='The path raw csv or tsv file')
    # parser.add_argument('--output', default='vaccine',
    #                     type=str, help='data source name')
    # args = parser.parse_args()

    # for is_binary in [True, False]:
    #     # on month
    #     if args.month:
    #         dataset = data_helper.load_data(args.month)
    #         # test on balanced data
    #         print('Test on balanced data')
    #         test_balance = cross_test_domain_clf(dataset, balance=True, binary=is_binary)
    #
    #         print('Test on unbalanced data')
    #         test_unbalance = cross_test_domain_clf(dataset, balance=False, binary=is_binary)
    #
    #         viz_perform(test_balance, './image/'+args.output+'/cross_clf_balance_month_'+str(is_binary)+'.png')
    #         viz_perform(test_unbalance, './image/'+args.output+'/cross_clf_unbalance_month_'+str(is_binary)+'.png')
    #
    #     # on year
    #     if args.year:
    #         dataset = data_helper.load_data(args.year)
    #         # test on balanced data
    #         print('Test on balanced data')
    #         test_balance = cross_test_domain_clf(dataset, balance=True, binary=is_binary)
    #
    #         print('Test on unbalanced data')
    #         test_unbalance = cross_test_domain_clf(dataset, balance=False, binary=is_binary)
    #
    #         viz_perform(test_balance, './image/'+args.output+'/cross_clf_balance_year_'+str(is_binary)+'.png')
    #         viz_perform(test_unbalance, './image/'+args.output+'/cross_clf_unbalance_year_'+str(is_binary)+'.png')

    file_list = [
        ('./data/vaccine/vaccine_month_sample.tsv', './data/vaccine/vaccine_year_sample.tsv', 'vaccine', 'Twitter data - vaccine'),
        ('./data/amazon/amazon_month_sample.tsv', './data/amazon/amazon_year_sample.tsv', 'amazon', 'Reviews data - music'),# './data/amazon/amazon_review_month_sample.tsv'
        ('./data/yelp/yelp_Hotels_month_sample.tsv', './data/yelp/yelp_Hotels_year_sample.tsv', 'yelp_hotel', 'Reviews data - hotels'),
        (None, './data/parties/parties_year_sample.tsv', 'parties', 'Politics - US political data'),
        ('./data/economy/economy_month_sample.tsv', './data/economy/economy_year_sample.tsv', 'economy', 'News data - economy'),
        ('./data/yelp/yelp_Restaurants_month_sample.tsv', './data/yelp/yelp_Restaurants_year_sample.tsv', 'yelp_rest', 'Reviews data - restaurants'), # './data/yelp/yelp_Restaurants_month_sample.tsv'
    ]
    for pair in file_list:
        print(pair)
        for is_binary in [False]: # True, skip binary currently
            # on month
            month_file = pair[0]
            year_file = pair[1]
            output = pair[2]

            if month_file:
                dataset = data_helper.load_data(month_file)
                # test on balanced data
                print('Test on balanced data')
                test_balance = cross_test_domain_clf(dataset, domain2month, data_name=None, balance=True, binary=is_binary)
                test_balance.to_csv('./tmp/' + output+ '_month.tsv', sep='\t')
                viz_perform(test_balance, pair[3],'./image/' + output + '/cross_clf_balance_month_' + str(is_binary) + '.pdf')
                test_balance = None

#                print('Test on unbalanced data')
#                test_unbalance = cross_test_domain_clf(dataset, domain2month, data_name=None, balance=False, binary=is_binary)
#                viz_perform(test_unbalance, pair[3], './image/'+output+'/cross_clf_unbalance_month_'+str(is_binary)+'.pdf')
#                test_unbalance = None
#                dataset = None
            # on year
            if year_file:
                dataset = data_helper.load_data(year_file)
                # test on balanced data
                print('Test on balanced data')
                test_balance = cross_test_domain_clf(dataset, domain2year, data_name=output, balance=True, binary=is_binary)
                test_balance.to_csv('./tmp/' + output+ '_year.tsv', sep='\t')
                viz_perform(test_balance, pair[3], './image/' + output + '/cross_clf_balance_year_' + str(is_binary) + '.pdf')
                test_balance = None

#                print('Test on unbalanced data')
#                test_unbalance = cross_test_domain_clf(dataset, domain2year, data_name=output, balance=False, binary=is_binary)
#                viz_perform(test_unbalance, pair[3], './image/'+output+'/cross_clf_unbalance_year_'+str(is_binary)+'.pdf')
                test_unbalance = None
