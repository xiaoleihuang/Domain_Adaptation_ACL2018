# step: build the vectorizer for year_month + general, f > 2, ngram = 3
#

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np
import pickle
from sklearn.metrics import classification_report, f1_score
from scipy.sparse import lil_matrix
from imblearn.over_sampling import RandomOverSampler
from multiprocessing import Pool


def data_batch_loader(
        data_name, test_time_label, file_type='year',
        batch_size=100, mode='train'):
    data_path = './data/'+data_name+'/'+data_name+'_'+file_type+'_sample.tsv'
    time_labels = sorted(
        [
            file_name.split('.')[0].split('_')[1].strip()
            for file_name in os.listdir('./vects1/' + data_name + '/')
            if file_type in file_name]
    )
    valid_time_label = time_labels[-3]

    if not test_time_label:
        test_time_label = time_labels[-2]  # the latest year

    batch_data = {
        'data': [], 'label': [], 'time_label': []
    }

    all_data = []
    all_label = []
    all_time_label = []

    with open(data_path) as datafile:
        datafile.readline()
        for line in datafile:
            infos = line.strip().split('\t')
            if mode == 'train' and infos[1] == test_time_label:
                continue
            if mode == 'test':
                if infos[1] != test_time_label:
                    continue
            if mode == 'valid':
                if infos[1] != valid_time_label:
                    continue
            all_data.append(infos[0])
            all_label.append(infos[2])
            all_time_label.append(infos[1])

    if mode == 'train':  # over sampling
        print('\t\tOver Sampling.......')
        sampler = RandomOverSampler(random_state=0)
        indices = [[item] for item in list(range(len(all_data)))]
        indices, all_label = sampler.fit_sample(indices, all_label)
        all_data = [all_data[item[0]] for item in indices]
        all_time_label = [all_time_label[item[0]] for item in indices]

    for item in zip(all_data, all_label, all_time_label):
        batch_data['data'].append(item[0])
        batch_data['label'].append(item[2])
        batch_data['time_label'].append(item[1])

        if len(batch_data['data']) >= batch_size:
            yield batch_data
            batch_data = {
                'data': [], 'label': [], 'time_label': [],
            }

    if len(batch_data['data']) > 0:
        yield batch_data


def create_domain_vects(data_name, mode='year'):
    data_path = './data/' + data_name + '/' + data_name + '_' + mode + '_sample.tsv'
    domain_docs = {'general': []}
    time_idx = 1

    # load the data for domain data
    print('\t\tLoading domain data')
    with open(data_path) as datafile:
        datafile.readline()
        for line in datafile:
            infos = line.strip().split('\t')
            domain_docs['general'].append(infos[0])

            if infos[time_idx] not in domain_docs:
                domain_docs[infos[time_idx]] = list()

            domain_docs[infos[time_idx]].append(infos[0])

    print('\t\tFitting domain data')
    for domain_name in domain_docs:
        print('\t\t\tWorking on: ' + domain_name)
        da_vect = TfidfVectorizer(min_df=2, ngram_range=(1, 3), stop_words='english')
        da_vect.fit(domain_docs[domain_name])
        pickle.dump(
            da_vect,
            open('./vects1/' + data_name + '/' + mode + '_' + str(domain_name) + '.pkl', 'wb')
        )
    return list(domain_docs.keys())


def create_domain_clfs(data_name, test_time_label, file_type='year'):
    domains = {file_type: []}
    sum_fea_size = 0
    fea_size = {file_type: dict()}

    # get feature size of each vectorizer:
    print('\t\tGet domain information.....')
    for file_name in os.listdir('./vects1/' + data_name + '/'):
        if file_type not in file_name:
            continue
        with open('./vects1/' + data_name + '/' + file_name, 'rb') as vect_pkl_f:
            vect_pkl = pickle.load(vect_pkl_f)
            cur_domain = file_name.split('.')[0].split('_')[1].strip()
            sum_fea_size += len(vect_pkl.vocabulary_)

            domains[file_type].append(cur_domain)
            fea_size[file_type][cur_domain] = len(vect_pkl.vocabulary_)

    print('Total feature size: ' + str(sum_fea_size))
    # load the time label: year by loop the file names in the vectorizer folder
    domains['year'] = sorted(domains['year'], reverse=True)  # reverse for set the 'general' in the 1st place
    domains['month'] = sorted(domains['month'])

    clf = SGDClassifier(
        loss='log', penalty='elasticnet', max_iter=2000,
        l1_ratio=0.1, n_jobs=-1, tol=0.0001)

    # load the data
    batch_size = 1000
    train_iter = data_batch_loader(
        data_name, test_time_label=test_time_label, file_type=file_type)

    # load the general vect
    general_vect = pickle.load(open('./vects1/' + data_name + '/' + file_type + '_general.pkl', 'rb'))
    print('\t\tBacth fit............')
    batch_count = 0
    for train_batch in train_iter:
        if len(np.unique(train_batch['label'])) == 1:
            continue

        print('Working on batch #' + str(batch_count))
        batch_count += 1
        # transform the data
        train_data = lil_matrix((len(train_batch['data']), sum_fea_size))
        train_data[:, :fea_size[file_type]['general']] = general_vect.transform(train_batch['data'])

        start_idx = fea_size['year']['general']

        for domain_name in domains[file_type]:
            if domain_name == 'general':
                continue
            with open('./vects1/' + data_name + '/' + file_type + '_' + str(domain_name) + '.pkl', 'rb') as vect_pkl_f:
                vect_pkl = pickle.load(vect_pkl_f)
                transformed_data = vect_pkl.transform(train_batch['data'])

                for label_idx in range(len(train_batch['time_label'])):
                    if train_batch['time_label'][label_idx] == domain_name:
                        train_data[label_idx, start_idx:start_idx + fea_size[file_type][domain_name]] = transformed_data[
                                                                                                   label_idx, :]
            start_idx += fea_size[file_type][domain_name]  # update the start index
        # partial training
        train_data = train_data.tocsr()

        clf.partial_fit(train_data, train_batch['label'], classes=['0', '1'])

    # save the clf
    print('\t\tSaving classifier............')
    with open('./clfs1/' + data_name + '_' + file_type + '.pkl', 'wb') as clf_file:
        pickle.dump(
            clf,
            clf_file
        )
    return clf


def run_exp(data_name, file_type, create_vects=False, create_clfs=False):
    print('Working on: ' + data_name + '..............................')
    if not os.path.exists('./vects1/' + data_name):
        os.mkdir('./vects1/' + data_name)

    if create_vects:
        print('\tCreating vects.........')
        domain_list = create_domain_vects(data_name, mode=file_type)
        print(domain_list)

    print('Creating logistic regression classifier------------')
    if create_clfs:
        clf = create_domain_clfs(data_name)
    else:
        clf = pickle.load(open('./clfs1/' + data_name + '.pkl', 'rb'))

    # only load general vectorizer
    gen_vect = pickle.load(open('./vects1/' + data_name + '/year_general.pkl', 'rb'))
    fea_size = clf.coef_.shape[1]  # feature size

    print('Validation.....')  # validation choose the 2nd latest year as the validation
    lambdas = [1, 10, 100, 200, 300]
    best_valid_f1 = 0
    best_lambda = 1

    for flip_l in lambdas:
        valid_iter = data_batch_loader(data_name, mode='valid')
        y_valids = []
        valid_preds = []

        for valid_batch in valid_iter:
            for label in valid_batch['label']:
                y_valids.append(label)
            valid_data = lil_matrix((len(valid_batch['data']), fea_size))
            valid_data[:, :len(gen_vect.vocabulary_)] = gen_vect.transform(valid_batch['data'])
            if flip_l != 1:
                valid_data = valid_data * flip_l
            predictions = clf.predict(valid_data)
            for label in predictions:
                valid_preds.append(label)
        tmp_f1 = f1_score(y_true=y_valids, y_pred=valid_preds, average='weighted')
        if tmp_f1 > best_valid_f1:
            best_valid_f1 = tmp_f1
            best_lambda = flip_l
            print(data_name + ' lambda: ' + str(best_lambda))
            print(data_name + ' valid f1: ' + str(best_valid_f1))

    print('Testing .....')
    test_iter = data_batch_loader(data_name, mode='test')
    y_preds = []
    y_truth = []

    print('Test by each batch')
    for test_batch in test_iter:
        for label in test_batch['label']:
            y_truth.append(label)

        # transform the test data:
        test_data = lil_matrix((len(test_batch['data']), fea_size))
        test_data[:, :len(gen_vect.vocabulary_)] = gen_vect.transform(test_batch['data'])

        # flip lambda
        test_data = test_data * best_lambda

        # prediction
        predictions = clf.predict(test_data)

        for label in predictions:
            y_preds.append(label)

    my_f1 = str(f1_score(y_true=y_truth, y_pred=y_preds, average='weighted'))
    my_report = classification_report(y_true=y_truth, y_pred=y_preds)

    print(data_name + '----- F1-score: ' + my_f1)
    with open('results.txt', 'a') as result_file:
        result_file.write('Working on ' + data_name + '--------------------\n')
        result_file.write(
            'Best valid result: ' + str(best_valid_f1) +
            ', lambda flip: ' + str(best_lambda) + '\n')
        result_file.write('F1: ' + my_f1 + '\n')
        result_file.write(my_report)
        result_file.write('\n----------------------------------------\n')


if __name__ == '__main__':
    data_list = [
        'amazon',
        'economy',
        'vaccine',
        'yelp_hotel',
        'yelp_rest',
        'parties',
    ]
    # multiprocess:
#    p = Pool(5)
#    p.map(run_exp, 'year')
#    p.map(run_exp, 'month')
    for file_type in ['year', 'month']:
        for data in data_list:
            run_exp(data, file_type=file_type, create_vects=False, create_clfs=False)
