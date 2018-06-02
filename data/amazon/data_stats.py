import pandas as pd
from collections import Counter
import sys


df = pd.read_csv(sys.argv[1], sep='\t')
print('Label Distribution Overall-----------')
print(Counter(df.label))
print('')
