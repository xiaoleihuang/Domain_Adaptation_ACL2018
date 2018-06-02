import pickle
import matplotlib.pyplot as plt
import sys
import argparse
from utils import model_helper

from sklearn.feature_selection import chi2, mutual_info_classif, f_classif

def by_chi2(fvs_da, y_trues, domain_vect, output='./feature_score/chi2.txt'):
    """
    Evaluate features by their chi-square
    :param fvs_da:
    :param y_trues:
    :param domain_vect:
    :param output:
    :return:
    """
    uniq_domains = domain_vect.uniq_domains
    uniq_domains.append('general')
    print(uniq_domains)

    # calculate chi2_score for each feature
    chi2s, pvals = chi2(fvs_da, y_trues)

    words = list(domain_vect.tfidf_vec_da['general'].vocabulary_.keys())
    start_idx = 0  # the feature idx counter

    for domain in uniq_domains:
        for word in words:
            if word in domain_vect.tfidf_vec_da[domain].vocabulary_:
                print(word + '\t' + domain + '\t' +
                      str(chi2s[start_idx+domain_vect.tfidf_vec_da[domain].vocabulary_[word]])
                      +'\t' + str(pvals[start_idx+domain_vect.tfidf_vec_da[domain].vocabulary_[word]]))
        start_idx = start_idx + len(domain_vect.tfidf_vec_da[domain].vocabulary_)


def by_mututal_info(fvs_da, y_trues, domain_vect, output='./feature_score/ftest.txt'):
    uniq_domains = domain_vect.uniq_domains
    uniq_domains.append('general')
    print(uniq_domains)

    # calculate chi2_score for each feature
    mi_vals = mutual_info_classif(fvs_da, y_trues)

    words = list(domain_vect.tfidf_vec_da['general'].vocabulary_.keys())
    start_idx = 0  # the feature idx counter

    for domain in uniq_domains:
        for word in words:
            if word in domain_vect.tfidf_vec_da[domain].vocabulary_:
                print(word + '\t' + domain + '\t' +
                      str(mi_vals[start_idx + domain_vect.tfidf_vec_da[domain].vocabulary_[word]]))
        start_idx = start_idx + len(domain_vect.tfidf_vec_da[domain].vocabulary_)


def by_anova(fvs_da, y_trues, domain_vect, output='./feature_score/mi.txt'):
    uniq_domains = domain_vect.uniq_domains
    uniq_domains.append('general')
    print(uniq_domains)

    # calculate chi2_score for each feature
    fvals, pvals = f_classif(fvs_da, y_trues)

    words = list(domain_vect.tfidf_vec_da['general'].vocabulary_.keys())
    start_idx = 0  # the feature idx counter

    for domain in uniq_domains:
        for word in words:
            if word in domain_vect.tfidf_vec_da[domain].vocabulary_:
                print(word + '\t' + domain + '\t' +
                      str(fvals[start_idx + domain_vect.tfidf_vec_da[domain].vocabulary_[word]])
                      + '\t' + str(pvals[start_idx + domain_vect.tfidf_vec_da[domain].vocabulary_[word]]))
        start_idx = start_idx + len(domain_vect.tfidf_vec_da[domain].vocabulary_)


def by_clf_weight(fvs_da, y_trues, domain_vect, output='./feature_score/weights.txt'):
    """
    This function will first train a classifier, then print out the trained weights
    :param fvs_da:
    :param y_trues:
    :return:
    """
    clf = model_helper.build_lr_clf()
    clf.fit(fvs_da, y_trues)
    # currently only supports binary classification,
    # because if multi-classes, the shape of coefficients will be changed
    words = list(domain_vect.tfidf_vec_da['general'].vocabulary_.keys())
    start_idx = 0  # the feature idx counter

    uniq_domains = domain_vect.uniq_domains
    uniq_domains.append('general')
    print(uniq_domains)

    for domain in uniq_domains:
        for word in words:
            if word in domain_vect.tfidf_vec_da[domain].vocabulary_:
                print(word + '\t' + domain + '\t' +
                      str(clf.coef_[0][start_idx + domain_vect.tfidf_vec_da[domain].vocabulary_[word]]))
        start_idx = start_idx + len(domain_vect.tfidf_vec_da[domain].vocabulary_)


def by_idf(domain_vect, fvs_da, output_path='./feature_score/idf.txt'):
    """

    :param domain_vect: Domain Adaption Vectorizer, Implementation of Sklearn's vectorizer
    :return: idf scores of each feature in each domain, for efficiency, it will return a generator
    """
    fvs_da = fvs_da.tocsc() # for faster column indexing
    tf_vectors = [0]*fvs_da.shape[1]

    for col_idx in range(fvs_da.shape[1]):
        non_zero_idx = fvs_da[:,col_idx].nonzero()[0][0]
        tf_vectors[col_idx] = fvs_da.getcol(col_idx)[non_zero_idx].toarray()[0][0]

    uniq_domains = domain_vect.uniq_domains
    uniq_domains.append('general')
    print(uniq_domains)
    feature_dicts = dict.fromkeys(domain_vect.tfidf_vec_da['general'].vocabulary_.keys(), dict.fromkeys(uniq_domains, 0.0))

    count_idx = 0 # the feature idx counter

    for domain in uniq_domains:
        for word in feature_dicts:
            # this is for tf-idf score
            if word in domain_vect.tfidf_vec_da[domain].vocabulary_:
                feature_dicts[word][domain] = tf_vectors[count_idx]

                print(word + '\t' + domain + '\t' + str(feature_dicts[word][domain]))
                count_idx += 1

            # if word in domain_vect.tfidf_vec_da[domain].vocabulary_:
                # feature_dicts[word][domain] = float(str(domain_vect.tfidf_vec_da[domain].idf_[
                #     domain_vect.tfidf_vec_da[domain].vocabulary_[word]])) # this is for idf_score
            # else:
            #     print(word + '\t' + domain +'\t' + str(0.0))
    # results = dict()
    # results['feature_dicts'] = feature_dicts
    # results['uniq_domains'] = uniq_domains
    # pickle.dump(feature_dicts, open(output_path, 'wb'))
    # return results

def extract_domain_idx(domain_vect, word):
    """

    :param domain_vect:
    :param word:
    :return: a list of word index pairs in each domain: word_idx, length of vocab in the domain
    """
    uniq_domains = domain_vect.uniq_domains
    uniq_domains.append('general')

    count = 0
    domain_word_idx = []
    for domain in uniq_domains:
        tmp_idx = domain_vect.tfidf_vec_da[domain].vocabulary_.get(word, -1)
        if tmp_idx == -1:
            domain_word_idx.append((domain, -1))
        else:
            domain_word_idx.append((domain, tmp_idx+count))
        count += len(domain_vect.tfidf_vec_da[domain].vocabulary_)

    return domain_word_idx

def extract_domain_weights(domain_idx, domain_clf):
    """
    Extract weights from classifier
    :param domain_idx:
    :param domain_clf:
    :return:
    """
    return [(w_idx[0], domain_clf.coef_[0][w_idx[1]]) if w_idx[1] != -1 else (w_idx[0], 0) for w_idx in domain_idx]

def visualize_word_feature(word, feature_dicts, mode='line_plot'):
    uniq_domains = feature_dicts['uniq_domains']
    if word in feature_dicts['feature_dicts']:
        features_x = [feature_dicts['feature_dicts'][word].get(domain, 0.0) for domain in uniq_domains]
        plt.plot([int(item) if item.isnumeric() else 2018 for item in uniq_domains], features_x)
        plt.title('Features of word: '+word)
        plt.show()
    else:
        print('Feature does not exist!')
        sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vect_path', default='./features/features_year.pkl',
                        type=str, help='The feature pickle path')
    parser.add_argument('--mode', default='by_idf',
                        type=str, help='Supports three modes: extract_feature; extract_weight; visualize') # supports two modes: visualize or extract word features or weights across domain
    parser.add_argument('--output_path', default='feature_dicts.dict',
                        type=str, help='the file to save data')
    parser.add_argument('--clf_path', default='./clf/clf_da_unbalance_month.pkl',
                        type=str, help='the classifier\'s file path')
    parser.add_argument('--feature_dict', default='feature_dicts.dict', type=str, help='the file path of feature dicts')
    args = parser.parse_args()

    vectorizer = pickle.load(open(args.vect_path, 'rb'))
    if args.mode == 'idf':
        by_idf(vectorizer['da_vect'], vectorizer['fvs_da'], args.output_path)
    elif args.mode == 'chi2': # by chi-square
        by_chi2(vectorizer['fvs_da'], vectorizer['label_raw'], vectorizer['da_vect'])
    elif args.mode == 'mi': # by mutual information
        by_mututal_info(vectorizer['fvs_da'], vectorizer['label_raw'], vectorizer['da_vect'])
    elif args.mode == 'anova': # by anova_test:
        by_anova(vectorizer['fvs_da'], vectorizer['label_raw'], vectorizer['da_vect'])
    elif args.mode == 'clf_weight':
        by_clf_weight(vectorizer['fvs_da'], vectorizer['label_raw'], vectorizer['da_vect'])
        # clf = pickle.load(open(args.clf_path, 'rb'))
        # vectorizer = pickle.load(open(args.vect_path, 'rb'))
        # word = input('Input the word you want to test: ').strip()
        # domain_idx = extract_domain_idx(vectorizer['da_vect'], word)
        # print(extract_domain_weights(domain_idx, clf))
    elif args.mode == 'visualize':
        feature_dicts = pickle.load(open(args.feature_dict, 'rb'))
        word = input('Input the word you want to visualize: ').strip()
        visualize_word_feature(word, feature_dicts)
    else:
        print('mode does not exist')
        sys.exit()