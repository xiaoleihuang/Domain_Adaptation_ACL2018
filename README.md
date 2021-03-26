# Domain_Adaptation_ACL2018
The repository of domain adaptation project for the [Examining Temporality in Document Classification](https://www.aclweb.org/anthology/P18-2110/) in ACL 2018. The slides will come soon.

# Table of Contents
 * Installation
 * Data
 * Usage
 * Contact and Citation

# Installation
 1. Platform:
  * Ubuntu 16.04
  * [Anaconda](https://www.anaconda.com/download), Python 3.6
 2. Run the followings to create environment:
  * `conda env create -f environment.yml`
  * `python -m nltk.downloader punkt stopwords`
  * `source activate domain`

# Data
 1. [Amazon CDs and Vinyl](http://jmcauley.ucsd.edu/data/amazon/)
 2. [Yelp reviews of Hotel and Restaurant](https://www.yelp.com/dataset)
 3. [Political Platforms](https://www.comparativeagendas.net/datasets_codebooks): Political Parties -> United States -> * Party Platform
 4. [Economical News](https://www.figure-eight.com/wp-content/uploads/2016/03/Full-Economic-News-DFE-839861.csv)
 5. [Vaccine Data](http://cmci.colorado.edu/~mpaul/resources.html)

# Usage
 1. Data extraction and sample:
  * Extraction: `python extract_data.py` within each data folder to extract data.
  * Sample: go to the utils folder, run `python under_sample.py`
 2. Run cross domain classification, under the project root folder: `python domain_clf_analysis.py`
 3. Generate the feature vectors: `python build_feas.py`
 4. Run grid search to find the optimal parameters: `python grid_search.py`
 5. Run domain adaptation (section **4.1** and **4.2** in the paper): `python run_exps.py`
 6. Combine both seasonal and non-seasonal information: `python build_sgd_base.py`

# Contact
<xiaolei.huang@colorado.edu>
