"""This script is used to tune parameters.
The data was splitted into training | validation | testing dataset,
the tuning process is only based on training and validation.


The results will be automatically write to folder grid search,
then it will write each experiment to a pickle file.
"""

from utils import data_helper, model_helper
import pickle
from sklearn.metrics.classification import f1_score
from imblearn.over_sampling import RandomOverSampler
from scipy.sparse import lil_matrix

if __name__ == '__main__':
    """Parameter Settings"""
    # l1, l2 is the proportion
    l2_only_solver = {'newton-cg', 'lbfgs', 'sag'}

    parameters_lr = {
        'C':[0.3, 1.0, 3.0], # smaller values specify stronger regularization.
        'solver':['liblinear'],
        'l1':[0, 0.3, 0.6, 1],  # when 0 means only l2, 1 means only l1
        'tol': [float('1e-4')],
        'balance':[False] # , False
    }

    lambdas = [0.3, 1, 3, 30, 300]

    # load data
    # data_paths = sys.argv[1:] # ['./features/features_month.pkl', './features/features_year.pkl']
    data_paths = [
        './features/vaccine/vaccine_month_tfidf.pkl',
        './features/vaccine/vaccine_year_tfidf.pkl',
        './features/amazon/amazon_review_month_tfidf.pkl',
        './features/amazon/amazon_review_year_tfidf.pkl',
        './features/yelp/yelp_Hotels_month_tfidf.pkl',
        './features/yelp/yelp_Hotels_year_tfidf.pkl',
        './features/yelp/yelp_Restaurants_month_tfidf.pkl',
        './features/yelp/yelp_Restaurants_year_tfidf.pkl',
        './features/parties/parties_year_tfidf.pkl',
        './features/economy/economy_rel_month_tfidf.pkl',
        './features/economy/economy_rel_year_tfidf.pkl',
        # binary
        './features/vaccine/vaccine_month_binary.pkl',
        './features/vaccine/vaccine_year_binary.pkl',
        './features/amazon/amazon_review_month_binary.pkl',
        './features/amazon/amazon_review_year_binary.pkl',
        './features/yelp/yelp_Hotels_month_binary.pkl',
        './features/yelp/yelp_Hotels_year_binary.pkl',
        './features/yelp/yelp_Restaurants_month_binary.pkl',
        './features/yelp/yelp_Restaurants_year_binary.pkl',
        './features/parties/parties_year_binary.pkl',
        './features/economy/economy_rel_month_binary.pkl',
        './features/economy/economy_rel_year_binary.pkl',
    ]

    # loop through each dataset
    for idx, dpath in enumerate(data_paths):
        print('Working on: '+dpath)
        print()
        # best performances
        best_da = {'params': None,
                   'clf': None,
                   'f1': 0.0}
        best_base = {'params': None,
                     'clf': None,
                     'f1': 0.0}

        print('Loading Data and Preprocessing')
        data_source = pickle.load(open(dpath, 'rb'))
        general_idx = -1 * len(data_source['da_vect'].tfidf_vec_da['general'].vocabulary_)

        # data splitting; skip the test idx, only tune paramters based on validation
        train_idx, valid_idx, _ = data_helper.shuffle_split_data(data_source['label_raw'])

        da_train = data_source['fvs_da'][train_idx]
        da_valid = data_source['fvs_da'][valid_idx]
        da_valid = lil_matrix(da_valid)
        da_valid[:][:general_idx] = 0

        base_train = data_source['fvs_base'][train_idx]
        base_valid = data_source['fvs_base'][valid_idx]
        y_da_train = data_source['label_raw'][train_idx]
        y_base_train = data_source['label_raw'][train_idx]
        y_valid = data_source['label_raw'][valid_idx]

        ros = RandomOverSampler(random_state=33) # fix the random for future reproductivity

        result_file = open('./grid_search/gr_'+str(dpath.split('/')[-1])+'.txt', 'w')
        result_file.write(dpath + '\n')
        result_file.write('\n')

        for is_bal in parameters_lr['balance']:
            if is_bal:
                da_train, y_da_train = ros.fit_sample(da_train, y_da_train)
                base_train, y_base_train = ros.fit_sample(base_train, y_base_train)
            for tol_val in parameters_lr['tol']:
                for c_val in parameters_lr['C']:
                    for l1_ratio in parameters_lr['l1']:
                        params = {
                            'C': c_val,
                            'l1_ratio': l1_ratio,
                            'tol':tol_val,
                            'n_job': -1, # to maximize using CPU
                            'bal': is_bal,
                            'max_iter': 2000
                        }
                        print(params)
                        if l1_ratio < 1 and l1_ratio > 0:
                            params['solver'] = 'sgd' # because the clf will be SGDClassifier
                            # build da clf
                            da_clf = model_helper.build_lr_clf(params)
                            da_clf.fit(da_train, y_da_train)
                            for lmda in lambdas:
                                params['lambda'] = lmda  # scale features

                                da_f1 = f1_score(y_true=y_valid, y_pred=da_clf.predict(da_valid.tocsr() * lmda),
                                                 average='weighted')
                                # save the best f1_score and params
                                if da_f1 > best_da['f1']:
                                    result_file.write('DA_F1: ' + str(da_f1))
                                    result_file.write(str(params))
                                    result_file.write('----------------------------------------\n\n')
                                    best_da['params'] = params
                                    best_da['clf'] = da_clf
                                    best_da['f1'] = da_f1


                            base_clf = model_helper.build_lr_clf(params)
                            base_clf.fit(base_train, y_base_train)
                            base_f1 = f1_score(y_true=y_valid, y_pred=base_clf.predict(base_valid), average='weighted')

                            if base_f1 > best_base['f1']:
                                result_file.write('BASE_F1: ' + str(base_f1))
                                result_file.write(str(params))
                                result_file.write('----------------------------------------\n\n')
                                best_base['params'] = params
                                best_base['clf'] = base_clf
                                best_base['f1'] = base_f1

                            print('Round: finished.')
                        else:
                            for solver in parameters_lr['solver']:
                                # filter out the solver that can not handle l1 regularizer
                                if l1_ratio == 1 and solver in l2_only_solver:
                                    continue

                                params['solver'] = solver
                                clf = model_helper.build_lr_clf(params)

                                da_clf = model_helper.build_lr_clf(params)
                                da_clf.fit(da_train, y_da_train)

                                for lmda in lambdas:
                                    da_f1 = f1_score(y_true=y_valid, y_pred=da_clf.predict(da_valid.tocsr()* lmda), average='weighted')
                                    # save the best f1_score and params
                                    if da_f1 > best_da['f1']:
                                        params['lambda'] = lmda  # scale features
                                        result_file.write('DA_F1: ' + str(da_f1))
                                        result_file.write(str(params))
                                        result_file.write('----------------------------------------\n\n')
                                        best_da['params'] = params
                                        best_da['clf'] = da_clf
                                        best_da['f1'] = da_f1
                                        print('DA F1: ' + str(da_f1))

                                base_clf = model_helper.build_lr_clf(params)
                                base_clf.fit(base_train, y_base_train)
                                base_f1 = f1_score(y_true=y_valid, y_pred=base_clf.predict(base_valid), average='weighted')


                                if base_f1 > best_base['f1']:
                                    result_file.write('BASE_F1: ' + str(base_f1))
                                    result_file.write(str(params))
                                    result_file.write('----------------------------------------\n\n')
                                    best_base['params'] = params
                                    best_base['clf'] = base_clf
                                    best_base['f1'] = base_f1

                                print('Round: finished.')

        result_file.flush()
        result_file.close()

        print(best_da['params'])
        print(best_da['f1'])
        print(best_base['params'])
        print(best_base['f1'])

        pickle.dump(best_da, open('./grid_search/da_'+str(dpath.split('/')[-1])+'.pkl', 'wb'))
        pickle.dump(best_base, open('./grid_search/base_'+str(dpath.split('/')[-1])+'.pkl', 'wb'))