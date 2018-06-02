import pickle, json
from random_domain import lr_domain_random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def da_tokenizer(text):
    return text.split()

file_list = [
    ('./data/amazon/amazon_review_month_sample.tsv', 'amazon_month'),
    ('./data/amazon/amazon_review_year_sample.tsv', 'amazon_year'),
    ('./data/yelp/yelp_Hotels_month_sample.tsv', 'yelp_hotel_month'),
    ('./data/yelp/yelp_Hotels_year_sample.tsv', 'yelp_hotel_year'),
    ('./data/yelp/yelp_Restaurants_month_sample.tsv', 'yelp_rest_month'),
    ('./data/yelp/yelp_Restaurants_year_sample.tsv', 'yelp_rest_year'),
]

pickle_dicts = dict()# save the paths of dataset to here

for data_path, name in file_list:
    print('Loading Data from '+data_path)
    dataset = pd.read_csv(data_path, sep='\t')
    save_dir = './tmp_outputs/'+name+'/'

    uniq_domains = dataset.time.unique()
    pickle_dicts[name] = {
        'uniq_domains':list(uniq_domains),
        'raw': os.path.abspath(data_path),
        'data': dict.fromkeys(uniq_domains, None),
        'da_vect': dict.fromkeys(uniq_domains, {'binary':None, 'tfidf':None}),
        'base_vect': {'binary':None, 'tfidf':None},
        'fvs_da': dict.fromkeys(uniq_domains, {'binary':None, 'tfidf':None}),
        'fvs_base':{'binary':None, 'tfidf':None},
    }
    pickle_dicts[name]['da_vect']['general'] = {'binary':None, 'tfidf':None}
    pickle_dicts[name]['fvs_da']['general'] = {'binary': None, 'tfidf': None}

    print()
    print(pickle_dicts[name])
    print()

    for dm in uniq_domains:
        print('Extract Subdomain data: '+str(dm))
        new_path = save_dir+str(dm) + '.tsv'
        dm_data = dataset[dataset.time == dm]
        dm_data.to_csv(new_path, header=True, sep='\t', index=False)

        dm_data = None
        pickle_dicts[name]['data'][dm] = os.path.abspath(new_path)

    # build vectorizer of base
    # binary or tfidf
    print('Extract features for base and general')
    for ftype in ['binary', 'tfidf']:
        if ftype == 'binary':
            base_vect = TfidfVectorizer(min_df=2, binary=True,
                                        use_idf=False, smooth_idf=False,
                                        ngram_range=(1,3))
        else:
            base_vect = TfidfVectorizer(min_df=2, ngram_range=(1,3))

        base_vect_path = save_dir + 'base_vect_' + ftype + '.pkl'
        fvs_base_path = save_dir + 'base_fvs_' + ftype + '.pkl'

        pickle_dicts[name]['base_vect'][ftype] = os.path.abspath(base_vect_path)
        pickle_dicts[name]['fvs_base'][ftype] = os.path.abspath(fvs_base_path)
        # base general just the same as the base
        pickle_dicts[name]['da_vect']['general'][ftype] = os.path.abspath(base_vect_path)
        pickle_dicts[name]['fvs_da']['general'][ftype] = os.path.abspath(fvs_base_path)

        print('Fitting starts')
        # fit data
        base_vect.fit(dataset.content)
        pickle.dump(base_vect, open(base_vect_path, 'wb')) # save the base vectorizer

        print('Feature converting')
        # convert to features
        fvs_base = base_vect.transform(dataset.content)
        pickle.dump(fvs_base, open(fvs_base_path, 'wb'))
        fvs_base = None
        base_vect = None

    dataset = None # save memory

    print('Extract features for each domain')
    # domain vectorizers start here
    for dpath_key in pickle_dicts[name]['data']:
        print('Domain: ' + str(dpath_key))
        da_data = pd.read_csv(pickle_dicts[name]['data'][dpath_key], sep='\t')

        for ftype in ['binary', 'tfidf']:
            if ftype == 'binary':
                da_vect = TfidfVectorizer(min_df=2, binary=True,
                    use_idf=False, smooth_idf=False, ngram_range=(1,3))
            else:
                da_vect = TfidfVectorizer(min_df=2, ngram_range=(1,3))

            # define paths
            da_vect_path = save_dir + 'da_vect_' + str(dpath_key) + '_' + ftype + '.pkl'
            fvs_da_path = save_dir + 'da_fvs_' + str(dpath_key) + '_' + ftype + '.pkl'
            pickle_dicts[name]['da_vect'][dpath_key][ftype] = os.path.abspath(da_vect_path)
            pickle_dicts[name]['fvs_da'][dpath_key][ftype] = os.path.abspath(fvs_da_path)

            # fit data
            da_vect.fit(da_data.content)
            pickle.dump(da_vect, open(da_vect_path, 'wb'))  # save the base vectorizer

            # convert to features
            fvs_da = da_vect.transform(da_data.content)
            pickle.dump(fvs_da, open(fvs_da_path, 'wb'))
            fvs_da = None
            da_vect = None

with open('large.pkl', 'wb') as writefile:
    pickle.dump(pickle_dicts, writefile)

