"""
This script is to add time column to the tagged data
"""
dataset = dict()
writefile = open('vaccine_year.csv', 'w')
with open('test_final_tagged_data.csv') as datafile:
    writefile.write(datafile.readline().strip() + ',Time\n')
    for line in datafile:
        line = line.strip()
        id = line.split(',')[1]
        if id not in dataset:
            dataset[id] = line

import glob
import json
from dateutil.parser import  parse as date_parser
filelist = glob.glob('/home/xiaolei/Documents/raw_data_vaccine/health*')
for filepath in filelist:
    with open(filepath) as datafile:
        print(filepath)
        for line in datafile:
            tmp_entity = json.loads(line)
            tmp_id = str(tmp_entity['id'])
            if tmp_id in dataset:
                writefile.write(dataset[tmp_id] + ',' + str(date_parser(tmp_entity['created_at']).month)+'\n')
                del dataset[tmp_id]

print(len(dataset))
#for id in dataset:
#    writefile.write(dataset[id] + ',2012.12\n')

writefile.flush()
writefile.close()
