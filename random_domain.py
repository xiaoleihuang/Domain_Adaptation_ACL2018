"""
This file train on random domains, then use the general features classify the test data
"""
from utils import data_helper, model_helper
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from scipy.sparse import lil_matrix

import warnings

warnings.simplefilter('ignore', UserWarning)

def lr_domain_random(results, balance=False, fs=False):
    """

    :param results:
    :param train_idx:
    :param test_idx:
    :param fs: feature selection
    :return:
    """
    label_raw = results['label_raw']
    fvs_base = results['fvs_base']
    fvs_da = results['fvs_da']

    train_idx, _, test_idx = data_helper.shuffle_split_data(label_raw)

    print('-----------------------Base Case--------------------------')
    # clf_base = LogisticRegression(class_weight='balanced')
    clf_base = model_helper.build_lr_clf()

    if balance:
        random_sampler = RandomOverSampler(random_state=0)
        train_data, train_label = random_sampler.fit_sample(fvs_base[train_idx], label_raw[train_idx])
    else:
        train_data, train_label = fvs_base[train_idx], label_raw[train_idx]

    # feature selection
    if fs:
        fea_selector = SelectKBest(mutual_info_classif, k=3000)
        train_data = fea_selector.fit_transform(train_data, train_label)
        test_data = fea_selector.transform(results['fvs_base'][test_idx])
    else:
        test_data = results['fvs_base'][test_idx]

    clf_base.fit(train_data, train_label)

    report_base = f1_score(y_true=results['label_raw'][test_idx],
                           y_pred=clf_base.predict(test_data),
                           average='weighted')
    print(report_base)

    print('-----------------------DA--------------------------')
    if balance:
        random_sampler = RandomOverSampler(random_state=0)
        train_data, train_label = random_sampler.fit_sample(fvs_da[train_idx], label_raw[train_idx])
    else:
        train_data, train_label = fvs_da[train_idx], label_raw[train_idx]
    # clf_da = LogisticRegression(class_weight='balanced')
    clf_da = model_helper.build_lr_clf()

    test_data = lil_matrix(results['fvs_da'][test_idx])
    general_len = -1*len(results['da_vect'].tfidf_vec_da['general'].vocabulary_)
    test_data[:, : general_len] = 0
    test_data = test_data.tocsc()

    # feature selection
    if fs:
        fea_selector = SelectKBest(mutual_info_classif, k=3000)
        train_data = fea_selector.fit_transform(train_data, train_label)
        fea_selector = SelectKBest(mutual_info_classif, k=3000)
        train_data = fea_selector.fit_transform(train_data, train_label)
        test_data = fea_selector.transform(test_data)

    clf_da.fit(train_data, train_label)

    report_da = f1_score(y_true=results['label_raw'][test_idx],
                         y_pred=clf_da.predict(test_data),
                         average='weighted')
    print(report_da)

    return report_base, report_da

if __name__ == '__main__':
    import pickle
    fs = [False]
    bals = [True, False]

    import sys

    for fea_sel in fs:
        for is_bal in bals:
            try:
                # print('Feature Selection Enabled: ' + str(fea_sel))
                print('Data is balanced: ' + str(is_bal))
                print('--------------------------year-------------------------------')
                datafile = pickle.load(open(sys.argv[1], 'rb'))
                lr_domain_random(datafile, balance=is_bal, fs=fea_sel)
                print('--------------------------month-------------------------------')
                datafile = pickle.load(open(sys.argv[2], 'rb'))
                lr_domain_random(datafile, balance=is_bal, fs=fea_sel)
            except Exception as e:
                continue
