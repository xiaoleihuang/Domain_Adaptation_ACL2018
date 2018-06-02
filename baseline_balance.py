import os
import sys
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

from sklearn.linear_model.base import BaseEstimator
from utils import data_helper, model_helper
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit

from sklearn.metrics import classification_report, f1_score
import numpy as np
np.random.seed(0)
from scipy.sparse import lil_matrix
import argparse
import pickle


def lr_cv(results, balance=False, clf_base=None, clf_da=None):
    """
    Cross-validation that compares DA method and non-DA method using Logistic Regression
    :param results:
    :param balance:
    :return:
    """

    label_raw = results['label_raw']
    fvs_base = results['fvs_base']
    fvs_da = results['fvs_da']

    # cross validation
    if balance:
        cv = StratifiedShuffleSplit(n_splits=5, test_size=.2, random_state=33)
        da_f1 = []
        base_f1 = []
        for train_idx, test_idx in cv.split(fvs_base, label_raw):
            random_sampler = RandomOverSampler(random_state=33)

            test_label = label_raw[test_idx]

            if not isinstance(clf_base, BaseEstimator):
                clf_base = model_helper.build_lr_clf()
            else:
                clf_base.warm_start = False # because we need to loop multiple times, each time requires a new instance of clf
            train_data, train_label = random_sampler.fit_sample(fvs_base[train_idx], label_raw[train_idx])
            clf_base.fit(train_data, train_label)
            pred = clf_base.predict(fvs_base[test_idx])
            base_f1.append(f1_score(y_pred=pred, y_true=test_label, average='weighted'))

            if not isinstance(clf_da, BaseEstimator):
                clf_da = model_helper.build_lr_clf()
            train_data, train_label = random_sampler.fit_sample(fvs_da[train_idx], label_raw[train_idx])
            clf_da.fit(train_data, train_label)
            pred = clf_da.predict(fvs_da[test_idx])
            da_f1.append(f1_score(y_pred=pred, y_true=test_label, average='weighted'))

        print('-----------------------Base Case--------------------------')
        print(str(np.mean(base_f1)))
        print('-----------------------DA--------------------------')
        print(str(np.mean(da_f1)))
    else:
        print('-----------------------Base Case--------------------------')
        if not isinstance(clf_base, BaseEstimator):
            clf_base = model_helper.build_lr_clf()
        scores = cross_val_score(clf_base, fvs_base, label_raw, cv=5, scoring='f1_weighted')
        print(np.mean(scores))


        print('-----------------------DA--------------------------')
        if not isinstance(clf_da, BaseEstimator):
            clf_da = model_helper.build_lr_clf()
        scores = cross_val_score(clf_da, fvs_da, label_raw, cv=5, scoring='f1_weighted')
        print(np.mean(scores))


def lr_tvt(results, balance=False, clf_da=None, clf_base=None):
    """
    The data was splitted into train | validation | testing dataset.
    :param results:
    :param balance:
    :return:
    """
    label_raw = results['label_raw']
    # label_encoder = results['label_encoder']
    # base_vect = results['base_vect']
    # da_vect = results['da_vect']
    fvs_base = results['fvs_base']
    fvs_da = results['fvs_da']

    train_idx, valid_idx, test_idx = data_helper.shuffle_split_data(label_raw)

    if balance:
        random_sampler = RandomOverSampler(random_state=33)
        da_train, da_label = random_sampler.fit_sample(fvs_da[train_idx], label_raw[train_idx])
        base_train, base_label = random_sampler.fit_sample(fvs_base[train_idx], label_raw[train_idx])

    else:
        base_train, base_label = fvs_base[train_idx], label_raw[train_idx]
        da_train, da_label = fvs_da[train_idx], label_raw[train_idx]

    print('-----------------------Base Case--------------------------')
    if not isinstance(clf_base, BaseEstimator):
        clf_base = model_helper.build_lr_clf()
    clf_base.fit(base_train, base_label)
    report = classification_report(y_true=label_raw[test_idx], y_pred=clf_base.predict(fvs_base[test_idx]))
    print(report)

    print('-----------------------DA--------------------------')
    if not isinstance(clf_da, BaseEstimator):
        clf_da = model_helper.build_lr_clf()
    clf_da.fit(da_train, da_label)
    report = classification_report(y_true=label_raw[test_idx], y_pred=clf_da.predict(fvs_da[test_idx]))
    print(report)


def lr_incre(results, mode='cv', balance=True, clf_da=None, clf_base=None):
    """
    logistic regression evaluation by incrementally adding dataset
    :param clf_base: the self-defined base classifier
    :param clf_da: the self-defined da classifier
    :param results:
    :param mode: default is cross validation, the other mode is to use train, validate, test
    :return:
    """
    scores = {'base':[], 'da':[]}
    start_point = int(len(results['label_raw']) * 0.1)
    step = int(len(results['label_raw'])*0.1)
    if mode == 'cv':
        for idx in range(start_point, len(results['label_raw'])+1, step):
            label_raw = results['label_raw'][:idx]
            label_base = label_raw[:]
            label_da = label_raw[:]

            fvs_base = results['fvs_base'][:idx]
            fvs_da = results['fvs_da'][:idx]

            if balance:
                random_sampler = RandomOverSampler(random_state=33)
                fvs_base, label_base = random_sampler.fit_sample(fvs_base, label_raw)
                fvs_da, label_da = random_sampler.fit_sample(fvs_da, label_raw)

            print('-----------------------Base Case--------------------------')
            if not isinstance(clf_base, BaseEstimator):
                clf_base = model_helper.build_lr_clf()
            score_base = np.mean(cross_val_score(clf_base, fvs_base, label_base, cv=5, scoring='f1'))
            print(score_base)
            scores['base'].append(str(score_base))

            print('-----------------------DA--------------------------')
            if not isinstance(clf_da, BaseEstimator):
                clf_da = model_helper.build_lr_clf()
            score_da = np.mean(cross_val_score(clf_da, fvs_da, label_da, cv=5, scoring='f1'))
            print(score_da)
            scores['da'].append(str(score_da))
    elif mode == 'tvt':
        train_idx, valid_idx, test_idx = data_helper.shuffle_split_data(results['label_raw'])
        for idx in range(start_point, len(results['label_raw'])+1, step):
            fvs_base = results['fvs_base'][train_idx][:idx]
            fvs_da = results['fvs_da'][train_idx][:idx]
            label_raw = results['label_raw'][train_idx][:idx]

            if balance:
                random_sampler = RandomOverSampler(random_state=33)
                train_da, da_label = random_sampler.fit_sample(fvs_da, label_raw)
                train_base, base_label = random_sampler.fit_sample(fvs_base, label_raw)
            else:
                train_da, da_label = fvs_da, label_raw
                train_base, base_label = fvs_base, label_raw
            print('-----------------------Base Case--------------------------')
            if not isinstance(clf_base, BaseEstimator):
                clf_base = model_helper.build_lr_clf()
            clf_base.fit(train_base, base_label)
            #, average='weighted'
            report_base = f1_score(y_true=results['label_raw'][test_idx], y_pred=clf_base.predict(results['fvs_base'][test_idx]), average='weighted')
            print(report_base)
            scores['base'].append(report_base)

            print('-----------------------DA--------------------------')
            if not isinstance(clf_da, BaseEstimator):
                clf_da = model_helper.build_lr_clf()
            clf_da.fit(train_da, da_label)
            report_da = f1_score(y_true=results['label_raw'][test_idx], y_pred=clf_da.predict(results['fvs_da'][test_idx]), average='weighted')
            print(report_da)
            scores['da'].append(report_da)
    else:
        print('Mode does not exist')
        sys.exit(-1)

    print(scores)
    return scores


def domain_idx_generator(results, dataset):
    for domain in results['da_vect'].uniq_domains:
        if domain == 'general':
            continue
        train_idx, test_idx = [], []
        [train_idx.append(idx) if item[-2] != int(domain) else test_idx.append(idx)
            for idx, item in enumerate(dataset)]
        yield train_idx, test_idx, domain
    # train_idx, test_idx = [], []
    # [train_idx.append(idx) if item[-2] != int(results['da_vect'].uniq_domains[-1]) else test_idx.append(idx)
    #     for idx, item in enumerate(dataset)]
    # print(results['da_vect'].uniq_domains[-1])
    # return train_idx, test_idx, results['da_vect'].uniq_domains[-1]


def lr_domain(results, train_idx, test_idx, balance=False, clf_da=None, clf_base=None, inits=None):
    """

    :param results:
    :param train_idx:
    :param test_idx:
    :return:
    """
    label_raw = results['label_raw']
    fvs_base = results['fvs_base']
    fvs_da = results['fvs_da']

    print('-----------------------Base Case--------------------------')
    if balance:
        random_sampler = RandomOverSampler(random_state=33)
        train_data, train_label = random_sampler.fit_sample(fvs_base[train_idx], label_raw[train_idx])
    else:
        train_data, train_label = fvs_base[train_idx], label_raw[train_idx]

    if not isinstance(clf_base, BaseEstimator):
        clf_base = model_helper.build_lr_clf()
        clf_base.fit(train_data, train_label)
        report_base = f1_score(y_true=results['label_raw'][test_idx],
                            y_pred=clf_base.predict(results['fvs_base'][test_idx]),
                            average='weighted')
#    report_base = '0.0'
    print(report_base)

    print('-----------------------DA--------------------------')
    if balance:
        random_sampler = RandomOverSampler(random_state=33)
        train_data, train_label = random_sampler.fit_sample(fvs_da[train_idx], label_raw[train_idx])
    else:
        train_data, train_label = fvs_da[train_idx], label_raw[train_idx]

    if not isinstance(clf_da, BaseEstimator):
        clf_da = model_helper.build_lr_clf(inits)
    clf_da.fit(train_data, train_label)

    # for using only general features
    general_len = -1 * len(results['da_vect'].tfidf_vec_da['general'].vocabulary_)
    test_data = lil_matrix(results['fvs_da'][test_idx])
    # print(test_data[0].toarray().mean())
    # f1 = test_data[0].toarray().mean()
    # print(general_len)
    # # # print(test_data.shape)

    # because the general features were appended finally, previous features are all domain features.
    test_data[:, :general_len] = 0  
    # print(test_data[0].toarray().mean())
    # # f2 = test_data[0].toarray().mean()

    # if inits and 'lambda' in inits:
    #     test_data = test_data * inits['lambda']
    # else:
    #     test_data = test_data*30
    test_data = test_data * 200

    report_da = f1_score(y_true=results['label_raw'][test_idx],
                         y_pred=clf_da.predict(test_data.tocsc()),
                         average='weighted')
    # print(classification_report(y_true=results['label_raw'][test_idx],
    #                      y_pred=clf_da.predict(test_data.tocsc()),))

    print(report_da)

    return report_base, report_da


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--fvs_path', default='./features/features_year.pkl', type=str)
    # parser.add_argument('--fea_type', default='tfidf', type=str, help='Type of features: binary or tfidf')
    # parser.add_argument('--split_mode', default='tvt', type=str)
    # parser.add_argument('--incre', default='domain', type=str)
    # parser.add_argument('--shuffle', default=False, type=bool)
    # parser.add_argument('--balance', default=False, type=bool)
    # parser.add_argument('--vizoutput', default='./image/eval/tmp.png', type=str)
    # args = parser.parse_args()
    # print(args)
    # print()
    #
    # if not os.path.exists(args.fvs_path):
    #     print('feature file does not exist, train a new file')
    #     data_path = input(
    #         "Input raw csv dataset ['./data/test_final_tagged_data_time.csv']: ")
    #     data_path = data_path or './data/test_final_tagged_data_time.csv'
    #     dataset = data_helper.load_data(data_path)
    #
    #     outp = data_helper.train_fvs_da(dataset, outputfile=args.fvs_path, balance=args.balance, fea_type=args.fea_type)
    #     print('New feature pickle file has been saved to: '+ str(outp))
    #     sys.exit(-1)
    #
    # try:
    #     results = pickle.load(open(args.fvs_path, 'rb'))
    # except IOError as ie:
    #     print('feature file can not be loaded')
    #     print(ie)
    #     sys.exit(-1)
    #
    # if args.shuffle:
    #     length = len(results['label_raw'])
    #     shuffle_indices = np.arange(length)
    #     np.random.shuffle(shuffle_indices)
    #
    #     results['fvs_base'] = results['fvs_base'][shuffle_indices]
    #     results['fvs_da'] = results['fvs_da'][shuffle_indices]
    #     results['label_raw'] = results['label_raw'][shuffle_indices]
    #
    # if args.incre == 'num':
    #     eval_result = lr_incre(results, args.split_mode, balance=args.balance)
    #     # plot
    #     x = [idx*1000 for idx in range(len(eval_result['base']))]
    #     plt.plot(x, eval_result['base'], label='base')
    #     plt.plot(x, eval_result['da'], label='da')
    #
    #     plt.legend(loc='best')
    #     plt.show()
    #
    # elif args.incre == 'domain':
    #     data_path = input(
    #         "Input raw dataset for domain ['./data/test_final_tagged_data_year.csv']: ")  # './data/test_final_tagged_data_time.csv'
    #     data_path = data_path or './data/test_final_tagged_data_year.csv'
    #     dataset = data_helper.load_data(data_path)
    #     report_base, report_da = {}, {}
    #
    #     for train_idx, test_idx, domain in domain_idx_generator(results, dataset):
    #         print("Domain:{2}\tTrain:{0}\tTest:{1}".format(len(train_idx), len(test_idx), domain))
    #         report_base[domain], report_da[domain] = lr_domain(results, train_idx, test_idx, balance=args.balance)
    #
    #     indx = list(range(len(report_da.keys())+1))
    #
    #     if 'month' in data_path:
    #         x = [int(item) for item in report_da.keys()]
    #         plt.bar([item -0.2 for item in x], list(report_base.values()), width=0.4, label='base')
    #         plt.bar(x, list(report_da.values()), width=0.4, label='da')
    #         # plt.xticks(indx, ['0']+[str(item) for item in report_da.keys()])
    #         plt.ylim([min(min(report_base.values()), min(report_da.values())) - 0.1,
    #                   max(max(report_base.values()), max(report_da.values())) + 0.05])
    #     else:
    #         x = [str(item) for item in report_da.keys()]
    #         plt.plot(x, list(report_base.values()), label='base')
    #         plt.plot(x, list(report_da.values()), label='da')
    #         plt.ylim([min(min(report_base.values()), min(report_da.values())) - 0.1,
    #                   max(max(report_base.values()), max(report_da.values())) + 0.05])
    #     print(sum(report_base.values())/len(report_base))
    #     print(sum(report_da.values())/len(report_da))
    #     plt.legend(loc='best')
    #     plt.show()
    # else:
    #     if args.split_mode == 'cv':
    #         eval_result = lr_cv(results, balance=args.balance)
    #     elif args.split_mode == 'tvt':
    #         eval_result = lr_tvt(results, balance=args.balance)
    #     else:
    #         print('split_mode does not exist, only cv or tvt')
    #         sys.exit(-1)

    file_list = [
#        ('./data/amazon/amazon_month_sample.tsv', './features/amazon/amazon_review_month_tfidf.pkl', {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 10}),
#        ('./data/economy/economy_month_sample.tsv', './features/economy/economy_rel_month_tfidf.pkl', {'C': 1, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 10}),
#        ('./data/vaccine/vaccine_month_sample.tsv', './features/vaccine/vaccine_month_tfidf.pkl', {'C': 1, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 10}),
#        ('./data/yelp/yelp_Hotels_month_sample.tsv', './features/yelp/yelp_Hotels_month_tfidf.pkl', {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 10}),
#        ('./data/yelp/yelp_Restaurants_month_sample.tsv', './features/yelp/yelp_Restaurants_month_tfidf.pkl', {'C': 1.0, 'l1_ratio': 0.1, 'tol': 0.001, 'n_job': -1, 'bal': False, 'max_iter': 2000, 'solver': 'liblinear', 'lambda': 10}),
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
        results = pickle.load(open(pair[1], 'rb'))
        data_path = pair[0]
        dataset = data_helper.load_data(data_path)
        report_base, report_da = {}, {}

        # for train_idx, test_idx, domain in domain_idx_generator(results, dataset):
        #     print("Domain:{2}\tTrain:{0}\tTest:{1}".format(len(train_idx), len(test_idx), domain))
        #     report_base[domain], report_da[domain] = lr_domain(results, train_idx, test_idx, balance=True, inits=pair[2])
        
        
        dig = domain_idx_generator(results, dataset)
        for train_idx, test_idx, domain in dig:
            if 'year' in pair[0]:
                if domain != sorted(results['da_vect'].uniq_domains)[-2]:
                    continue
            print("Domain:{2}\tTrain:{0}\tTest:{1}".format(len(train_idx), len(test_idx), domain))
            report_base[domain], report_da[domain] = lr_domain(results, train_idx, test_idx, balance=True, inits=pair[2])


        # indx = list(range(len(report_da.keys())+1))
        #
        # if 'month' in data_path:
        #     x = [int(item) for item in report_da.keys()]
        #     plt.bar([item -0.2 for item in x], list(report_base.values()), width=0.4, label='base')
        #     plt.bar(x, list(report_da.values()), width=0.4, label='da')
        #     # plt.xticks(indx, ['0']+[str(item) for item in report_da.keys()])
        #     plt.ylim([min(min(report_base.values()), min(report_da.values())) - 0.1,
        #               max(max(report_base.values()), max(report_da.values())) + 0.05])
        # else:
        #     x = [str(item) for item in report_da.keys()]
        #     plt.plot(x, list(report_base.values()), label='base')
        #     plt.plot(x, list(report_da.values()), label='da')
        #     plt.ylim([min(min(report_base.values()), min(report_da.values())) - 0.1,
        #               max(max(report_base.values()), max(report_da.values())) + 0.05])
        print(sum(report_base.values())/len(report_base))
        print(sum(report_da.values())/len(report_da))
#        plt.legend(loc='best')
#        plt.show()
