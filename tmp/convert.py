import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd


def viz_perform(df, title, outpath='./image/output.pdf'):
    """
    Heatmap visualization
    :param df: an instance of pandas DataFrame
    :return:
    """
    a4_dims = (14, 10)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.set(font_scale=2)
    viz_plot = sns.heatmap(df, annot=True, cbar=False, annot_kws={"size": 30}, cmap="YlGnBu", vmin=df.values.min(), fmt='.3f')
    plt.yticks(rotation=30, fontsize=30)
    plt.xticks(rotation=20, fontsize=30)
    plt.xlabel('Train', fontsize=40)
    plt.ylabel('Test', fontsize=40)
    plt.title(title, fontsize=40)
    plt.tight_layout()
    viz_plot.get_figure().savefig(outpath, format='pdf')
    plt.close()

file_list = [('yelp_rest_year.tsv', 'Reviews data - restaurants', 'year_rest'),
 ('yelp_hotel_year.tsv', 'Reviews data - hotels', 'year_hotel'),
 ('amazon_month.tsv', 'Reviews data - music', 'month_amazon'),
 ('economy_month.tsv', 'News data - economy', 'month_economy'),
 ('vaccine_year.tsv', 'Twitter data - vaccine', 'year_vaccine'),
 ('amazon_year.tsv', 'Reviews data - music', 'year_amazon'),
 ('vaccine_month.tsv', 'Twitter data - vaccine', 'month_vaccine'),
 ('parties_year.tsv', 'Politics - US political data', 'year_parties'),
 ('yelp_hotel_month.tsv', 'Reviews data - hotels', 'month_hotel'),
 ('economy_year.tsv', 'News data - economy', 'year_economy'),
 ('yelp_rest_month.tsv', 'Reviews data - restaurants', 'month_rest')]
for pair in file_list:
    if pair[0].endswith('.tsv'):
        df = pd.read_csv(pair[0], sep='\t', index_col=0)
        viz_perform(df, title=pair[1], outpath=pair[2] + '.pdf')
    
