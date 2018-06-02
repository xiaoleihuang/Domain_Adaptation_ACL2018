"""
This script is to extract data from tea parties,
the time attribute only year available.
"""
import pandas as pd
from nltk.tokenize import word_tokenize


def prep_doc(review_text):
    return " ".join([word.strip() for word in word_tokenize(review_text.strip().lower()) if len(word.strip())>0])


def time2num(year):
    """
    Convert year 2 labels
    :param year:
    :return:
    """
    # time1 = [1948, 1952, 1956, 1960, 1964]
    # time2 = [1968, 1972, 1976, 1980, 1984, ]
    # time3 = [1988, 1992, 1996, 2000, 2004]
    # time4 = [2008, 2012, 2016]
    time1 = [1948, 1952, 1956] # Eisenhower and truman: mixed
    time2 = [1960, 1964, 1968] # kennedy and johnson: democratic
    time3 = [1972, 1976, 1980] # carter and ford, nixon: mixed
    time4 = [1984, 1988, 1992] # reagan and bush: republican
    time5 = [1996, 2000, 2004] # clinton and bush: mixed
    time6 = [2008, 2012, 2016] # obama democratic
    if year in time1:
        return 1
    elif year in time2:
        return 2
    elif year in time3:
        return 3
    elif year in time4:
        return 4
    elif year in time5:
        return 5
    elif year in time6:
        return 6


def parties2csv(d_path, r_path, output='parties_year.tsv'):
    with open(output, 'w') as writefile:
        writefile.write('\t'.join(['content', 'time', 'label']) + '\n')
        col_names = ['year', 'description']
        for tmp_path in [r_path, d_path]:
            datafile = pd.read_csv(tmp_path)
            datafile = datafile[col_names]
            datafile['description'] = datafile['description'].apply(lambda x:prep_doc(str(x)))

            if 'republican' in tmp_path:
                label = 0
            else:
                label = 1

            for _, row in datafile.iterrows():
                writefile.write(row['description']+'\t'+str(time2num(row['year']))+'\t'+str(label)+'\n')


if __name__ == '__main__':
    """
    python extract_data.py democratic_party_platform.csv republican_party_platform.csv
    """
    import sys
    parties2csv(sys.argv[1], sys.argv[2])
