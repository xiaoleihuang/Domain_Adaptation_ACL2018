import pickle
import matplotlib.pyplot as plt
import sys, os
import argparse
import numpy as np
import operator
import textwrap

from utils import data_helper

def by_feature_value(feature_dicts, topn=30, mode='by_domain', if_reverse=True, save_path='value_word_bydomain.pkl'):
    """
    extract the top n ranking features, where the features are ranked by weight (p_m - p_general)/p_g.
    :param feature_dicts:
    :param topn:
    :param if_reverse
    :param mode: supports two modes, by_domain separated rankings per domain, or all no separations
    :return: a dictionary of ranking features, where the key is the weight and value is a list of words which share the same weight.
    """
    value_word = dict()
    if mode == 'by_domain':
        # extract uniq_domains
        uniq_domains = list(feature_dicts[list(feature_dicts.keys())[0]].keys())
        uniq_domains.remove('general')
        for domain in uniq_domains:
            value_word[domain] = dict()
            for word in feature_dicts:
                if feature_dicts[word]['general'] == 0:
                    continue
                value_word[domain][word] = abs(feature_dicts[word][domain] - feature_dicts[word]['general'])/abs(feature_dicts[word]['general'])
                # rank and select the topn

            tmp_pair = list()
            value_word[domain] = sorted(value_word[domain].items(), key=operator.itemgetter(1), reverse=if_reverse)[:int(topn)]
            # value_word[domain] = [(word, value_word[domain][word]) for word in sorted(value_word[domain], key=lambda x:value_word[domain][x], reverse=if_reverse)[:topn]]

    elif mode == 'all':
        for word in feature_dicts:
            max_value = ['', -float("inf")]
            for domain in feature_dicts[word]:
                if domain == 'general':
                    continue
                if feature_dicts[word]['general'] == 0:
                    continue

                tmp_value = (feature_dicts[word][domain] - feature_dicts[word]['general'])/feature_dicts[word]['general']
                if tmp_value > max_value[1]:
                    max_value[0] = word+'#'+str(domain)
                    max_value[1] = tmp_value
            value_word[max_value[0]] = max_value[1]

        value_word = [(word, value_word[word]) for word in sorted(value_word, key=lambda x:value_word[x], reverse=if_reverse)[:topn]]
    else:
        print('Input Mode Does not exist')
        sys.exit()
    pickle.dump(value_word, open(save_path, 'wb'))
    return value_word


def create_barh(ax_plt, word_value_list, title=''):
    y_pos = np.arange(len(word_value_list))
    words = ['\n'.join(textwrap.wrap(item[0], 15)) for item in word_value_list]
    values = [item[1] for item in word_value_list]
    ax_plt.barh(y_pos, values, align='center', color='blue')
    ax_plt.set_yticks(y_pos)
    ax_plt.set_yticklabels(words)
    ax_plt.invert_yaxis() # labels read top-to-bottom
    if len(title) > 0:
        ax_plt.set_xlabel(title)


def viz_ranked_features(value_word):
    """

    :param value_word:
    :return:
    """
    # check the type of value_word, if it's dict, it was for domain by domain, otherwise it was for whole
    size = 1
    if type(value_word) is dict:
        for domain in value_word:
            fig, ax = plt.subplots()
            fig.set_figheight(6)
            fig.set_figwidth(12)
            create_barh(ax, value_word[domain], str(domain))
            fig.savefig('./image/'+str(domain)+'.png')
    else:
        fig, ax = plt.subplots()
        fig.set_figheight(8)
        fig.set_figwidth(12)
        create_barh(ax, value_word, 'All Domains Together')
        fig.savefig('./image/all.png')
    # plt.show()


if __name__ == '__main__':
    """
    Usage: python rank_feature.py --fd_path ./feature_score/year/clf_weight.txt --mode all 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd_path', default='./feature_dicts_year.txt', type=str,
                        help = 'the file path of feature dicts')
    parser.add_argument('--mode', default='by_domain', type=str,
                        help='Supports two modes: extract_feature; extract_weight; visualize')  # supports two modes: visualize or extract word features or weights across domain
    parser.add_argument('--viz', default=True, type=bool,
                        help='Whether visualize the ranked features')
    args = parser.parse_args()

    fd_dict = data_helper.load_feature(args.fd_path)
    vw_dict = by_feature_value(fd_dict, 10, args.mode, if_reverse=True, save_path='./feature_score/vw_'+str(args.mode)+'.pkl')

    if args.viz:
        viz_ranked_features(vw_dict)
