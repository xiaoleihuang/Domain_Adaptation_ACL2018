import numpy as np
np.random.seed(0)
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import lil_matrix, csc_matrix, hstack
from sklearn.linear_model import SGDClassifier, LogisticRegression
import pickle


class DomainVectorizer_tfidf(TransformerMixin):
    def __init__(self):
        self.uniq_domains = None
        self.tfidf_vec_da = None
        self.use_large = False

    def da_tokenizer(self, text):
        return text.split()

    def fit(self, dataset, y=None):
        # if len(dataset) > 15469:  # this number is length of "./yelp/yelp_Hotels_year_sample.tsv"
        #     self.use_large = True
        print('start to fit')

        self.uniq_domains = sorted(np.unique([item[-2] for item in dataset]))
        self.tfidf_vec_da = dict.fromkeys(self.uniq_domains)
        if not self.use_large:
            for key in self.tfidf_vec_da:
                print('Domain:' + str(key))
                self.tfidf_vec_da[key] = TfidfVectorizer(ngram_range=(1, 3), min_df=2)
                self.tfidf_vec_da[key].fit([item[-3] for item in dataset if item[-2] == key])
            self.tfidf_vec_da["general"] = TfidfVectorizer(ngram_range=(1, 3), min_df=2)
            self.tfidf_vec_da["general"].fit([item[-3] for item in dataset])
        else:
            for key in self.tfidf_vec_da:
                print('Domain:' + str(key))
                self.tfidf_vec_da[key] = \
                    '/home/xiaolei/Documents/Github/Domain_Adaptation/features/extra/' + str(key) + '.pkl'
                tmp_vect = TfidfVectorizer(min_df=2, tokenizer=self.da_tokenizer)
                tmp_vect.fit([item[-3] for item in dataset if item[-2] == key])
                pickle.dump(tmp_vect, open(self.tfidf_vec_da[key], 'wb'))

            tmp_vect = TfidfVectorizer(min_df=2, tokenizer=self.da_tokenizer)
            self.tfidf_vec_da["general"] = \
                '/home/xiaolei/Documents/Github/Domain_Adaptation/features/extra/general.pkl'
            tmp_vect.fit([item[-3] for item in dataset])
            pickle.dump(tmp_vect, open(self.tfidf_vec_da['general'], 'wb'))
        print('finished fitting')
        return self

    def transform(self, dataset):
        fvs = csc_matrix(np.zeros(shape=(len(dataset), 1)))

        if not self.use_large:
            for domain in self.uniq_domains:
                tmp_fvs = csc_matrix(self.tfidf_vec_da[domain].transform(
                    [item[-3] if item[-2] != domain else "" for item in dataset]
                    # [item[-3] for item in dataset]
                ))
                fvs = hstack([fvs, tmp_fvs])
            fvs = fvs[:, 1:]
            tmp_fvs = csc_matrix(self.tfidf_vec_da['general'].transform(
                [item[-3] for item in dataset])
            )
            fvs = hstack([fvs, tmp_fvs])
        else:
            fvs = csc_matrix(np.zeros(shape=(len(dataset), 1)))

            for domain in self.uniq_domains:
                dm_vect = pickle.load(open(self.tfidf_vec_da[domain], 'rb'))
                tmp_fvs = csc_matrix(dm_vect.transform(
                    # [item[-3] if item[-2] != domain else "" for item in dataset]
                    [item[-3] for item in dataset]
                ))
                print(tmp_fvs.shape)
                print(fvs.shape)
                fvs = hstack([fvs, tmp_fvs])
            fvs = fvs[:, 1:]

            dm_vect = pickle.load(open(self.tfidf_vec_da['general'], 'rb'))
            tmp_fvs = csc_matrix(dm_vect.transform(
                [item[-3] for item in dataset]))
            fvs = hstack([fvs, tmp_fvs])
        return fvs


def build_lr_clf(init_settings=None):
    """Create Customized logistic classifiers, especially for using elastic-net as regularization

    """
    if not init_settings:
        # init_settings = {  # binary vaccine
        #     'C': 2, 'l1_ratio': 0.1, 'tol': 1e-04,
        #     'n_job': -1, 'bal': True, 'max_iter': 2000,
        #     'solver': 'liblinear'}
        init_settings = { # binary vaccine
            'C': 1, 'l1_ratio': 0.0, 'tol': 1e-04,
            'n_job': -1, 'bal': True,
            'solver': 'liblinear'}
        # init_settings = {  # binary vaccine
        #     'C': 3, 'l1_ratio': 0.6, 'tol': 1e-05,
        #     'n_job': -1, 'bal': True, 'max_iter': 3000,
        #     'solver': 'liblinear'}
    # return LogisticRegression(class_weight='balanced')
    if init_settings['l1_ratio'] < 1 and init_settings['l1_ratio'] > 0:
        return SGDClassifier(loss='log', penalty='elasticnet', class_weight='balanced',
                             l1_ratio= init_settings['l1_ratio'],
                             alpha=0.0001*init_settings['C'], tol=init_settings['tol'],
                             n_jobs=-1, max_iter=2000)
    else:
        if init_settings['l1_ratio'] == 0:
            return LogisticRegression(penalty='l2', class_weight='balanced',
                                      solver=init_settings['solver'], tol=init_settings['tol'],
                                      C=init_settings['C'], n_jobs=-1)
        else:
            return LogisticRegression(penalty='l1', class_weight='balanced',
                                      solver=init_settings['solver'], tol=init_settings['tol'],
                                      C=init_settings['C'], n_jobs=-1)


class DomainVectorizer_binary(TransformerMixin):
    """
    This script is to vectorize text to binary features.
    """
    def __init__(self):
        self.uniq_domains = None
        self.tfidf_vec_da = None
        self.use_large = False

    def da_tokenizer(self, text):
        return text.split()

    def fit(self, dataset, y=None):
        # if len(dataset) > 15469:
        #     self.use_large = True
        print('start to fit')

        self.uniq_domains = sorted(np.unique([item[-2] for item in dataset]))
        self.tfidf_vec_da = dict.fromkeys(self.uniq_domains)
        if not self.use_large:
            for key in self.tfidf_vec_da:
                print('Domain:' + str(key))
                self.tfidf_vec_da[key] = TfidfVectorizer(ngram_range=(1,3),
                    binary=True, use_idf=False, smooth_idf=False, min_df=2)
                self.tfidf_vec_da[key].fit([item[-3] for item in dataset if item[-2] == key])
            self.tfidf_vec_da["general"] = TfidfVectorizer(ngram_range=(1,3),
                binary=True, use_idf=False, smooth_idf=False, min_df=2)
            self.tfidf_vec_da["general"].fit([item[-3] for item in dataset])
        else:
            for key in self.tfidf_vec_da:
                print('Domain:' + str(key))
                self.tfidf_vec_da[key] = \
                    '/home/xiaolei/Documents/Github/Domain_Adaptation/features/extra/' \
                    + str(key) + '_binary.pkl'
                tmp_vect = TfidfVectorizer(min_df=2, binary=True, token_pattern=None,
                    tokenizer = self.da_tokenizer, use_idf=False, smooth_idf=False)
                tmp_vect.fit([item[-3] for item in dataset if item[-2] == key])
                pickle.dump(tmp_vect, open(self.tfidf_vec_da[key], 'wb'))

            tmp_vect = TfidfVectorizer(min_df=2, binary=True, use_idf=False,
                    tokenizer=self.da_tokenizer, smooth_idf=False, token_pattern=None)
            self.tfidf_vec_da["general"] = \
                '/home/xiaolei/Documents/Github/Domain_Adaptation/features/extra/general_binary.pkl'
            tmp_vect.fit([item[-3] for item in dataset])
            pickle.dump(tmp_vect, open(self.tfidf_vec_da['general'], 'wb'))
        print('finished fitting')
        return self

    def transform(self, dataset):
        fvs = csc_matrix(np.zeros(shape=(len(dataset), 1)))

        if not self.use_large:
            for domain in self.uniq_domains:
                tmp_fvs = csc_matrix(self.tfidf_vec_da[domain].transform(
                    [item[-3] if item[-2] != domain else "" for item in dataset]
                    # [item[-3] for item in dataset]
                ))
                fvs = hstack([fvs, tmp_fvs])
            fvs = fvs[:, 1:]
            tmp_fvs = csc_matrix(self.tfidf_vec_da['general'].transform(
                [item[-3] for item in dataset])
            )
            fvs = hstack([fvs, tmp_fvs])
        else:
            fvs = csc_matrix(np.zeros(shape=(len(dataset), 1)))

            for domain in self.uniq_domains:
                dm_vect = pickle.load(open(self.tfidf_vec_da[domain], 'rb'))
                tmp_fvs = csc_matrix(dm_vect.transform(
                    # [item[-3] if item[-2] != domain else "" for item in dataset]
                    [item[-3] for item in dataset]
                ))
                print(tmp_fvs.shape)
                print(fvs.shape)
                fvs = hstack([fvs, tmp_fvs])
            fvs = fvs[:, 1:]

            dm_vect = pickle.load(open(self.tfidf_vec_da['general'], 'rb'))
            tmp_fvs = csc_matrix(dm_vect.transform(
                [item[-3] for item in dataset]))
            fvs = hstack([fvs, tmp_fvs])
        return fvs


def myloss(y_true, y_pred, weights):
    """Customized loss function
	"""
    from keras import backend as K
    return K.mean(K.square(y_pred - y_true), axis=-1) + K.sum(0.001 * K.square(weights))
