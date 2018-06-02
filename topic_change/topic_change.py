import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pickle


def build_dict_idx(data_name):
    # read data from the raw data tsv file
    doc_clean = []
    with open('../raw_tsv_data/'+data_name+'.tsv') as data_file:
        data_file.readline()
        for line in data_file:
            infos = line.strip().split('\t')
            doc_clean.append(infos[0].split())
    dictionary = Dictionary(doc_clean)
    dictionary.save('./dict/'+data_name+'.pkl')

    # convert the doc as indices
    doc_matrix = np.asarray([dictionary.doc2bow(doc) for doc in doc_clean])
    np.save(open('./dict/'+data_name+'.npy', 'wb'), doc_matrix)


def build_lda_model(data_name):
    # load the training data
    train_data = np.load(open('./dict/'+data_name+'.npy', 'rb'))
    # load the dictionary
    dictionary = pickle.load(open('./dict/'+data_name+'.pkl', 'rb'))
    lda = LdaModel(train_data,
        id2word=dictionary, num_topics=20,
        passes=2, alpha='symmetric', eta=None)
    
    lda.print_topics(num_topics=20, num_words=10)

    # save the model
    lda.save('./lda/'+data_name+'.model')


def run_exp(data_name, build_dict=True, build_lda=True):
    print('Working on: ' + data_name)
    if build_dict:
        print('Building dictionary........')
        build_dict_idx(data_name)
    dictionary = pickle.load(open('./dict/'+data_name+'.pkl', 'rb'))
    
    if build_lda:
        print('Building lda model....')
        build_lda_model(data_name)
    lda_model = LdaModel.load('./lda/'+data_name+'.model')

    count_num = dict()
    with open('../raw_tsv_data/'+data_name+'.tsv') as data_file:
        data_file.readline()
        for line in data_file:
            infos = line.strip().split('\t')
            if infos[1] not in count_num:
                count_num[infos[1]] = [0] * 20
            topic_doc = lda_model[dictionary.doc2bow(infos[0].strip().split())]
            topic_num = sorted(topic_doc, key=lambda x:x[1])[-1][0]
            count_num[infos[1]][topic_num] = count_num[infos[1]][topic_num] + 1
    # normalize
    for key in count_num:
        sum_value = sum(count_num[key])
        new_dict = dict()
        for idx in range(len(count_num[key])):
#            new_dict['topic'+str(idx)] = count_num[key][idx]/sum_value
            count_num[key][idx] = count_num[key][idx]/sum_value
#        count_num[key] = new_dict
    
    # visualize
    bar_list = []
    topic_domain = dict()

    topic_num = 20
    ind = np.arange(topic_num)
    prev_stack_vals = [0] * topic_num
    width = 0.35
    for idx in range(topic_num):  # topic number
        bar_val = []
        for key in count_num:
            bar_val.append(count_num[key][idx])
        topic_domain['t'+str(idx)] = bar_val
#        bar_val = []
#        for key in count_num:
#            bar_val.append(count_num[key][idx])
#            prev_stack_vals[idx] = prev_stack_vals[idx] + count_num[key][idx]
#        bar_tmp = plt.bar(ind, bar_val, bottom = prev_stack_vals)
#        bar_list.append(bar_tmp)
    print(topic_domain)
    topic_domain = pd.DataFrame(topic_domain)
    topic_domain.plot.bar(stacked=True, legend=False)
    plt.savefig('./pic/'+data_name+'.pdf')
#    plt.ylabel('Topic Percentage')
#    plt.title('Time Domains')
#    plt.xticks(ind, count_num.keys())
#    plt.show()

if __name__ == '__main__':
    data_list = [
        ('vaccine', 'vaccine_year'),
        ('amazon', 'amazon_month'),
        ('amazon', 'amazon_year'),
#        ('dianping', 'dianping_month'),
#        ('dianping', 'dianping_year'),
#        ('google', 'economy_month'),
#        ('google', 'economy_year'),
#        ('google', 'parties_year'),
        ('vaccine', 'vaccine_month'),
#        ('yelp_hotel', 'yelp_hotel_month'),
#        ('yelp_hotel', 'yelp_hotel_year'),
#        ('yelp_rest', 'yelp_rest_month'),
#        ('yelp_rest', 'yelp_rest_year'),
    ]

    for pair in data_list:
        run_exp(pair[1])
