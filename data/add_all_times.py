"""
This script is to add month label (source) to the year corpus (target)
"""
import sys

target_path = sys.argv[1]
source_path = sys.argv[2]
output_name = sys.argv[3] # such as 'amazon'

# load the target data
# extract the source to the dictionary: content - time label
source_dict = dict()
with open(source_path) as source_file:
    source_file.readline()
    for line in source_file:
        infos = line.strip().split('\t')
        source_dict[infos[0].strip()] = infos[1].strip()

count = 0
# loop through the target data
with open(output_name+'_year_month.tsv', 'w') as write_file:
    with open(target_path) as target_file:
        write_file.write(target_file.readline().strip()+'\t'+'time1\n')
        for line in target_file:
            infos = line.strip().split('\t')
            if infos[0].strip() not in source_dict:
                count += 1
                print(infos[0].strip())
            else:
                write_file.write(line.strip() + '\t' + source_dict[infos[0].strip()] + '\n')
print(count)
