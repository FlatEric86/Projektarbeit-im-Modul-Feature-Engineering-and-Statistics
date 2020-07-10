import os
import numpy as np

# Grenzwert ab welchem ein Teildatensatz relevant ist und aggregiert wird
treshold_val = 4150



def csv_to_list(fname, mod=None):

    date_col = []
    data_col = []

    with open('./resold_analysis/' + fname) as fin:
        for line in fin.readlines():

            line = line.strip('\n').split('\t')

            date_col.append(line[0])
            data_col.append(int(line[-1]))

    if mod == None:
        return data_col

    else:
        return date_col, data_col
            


data_colms = []
ID         = []

flag = 0
for fname in os.listdir('./resold_analysis'):
    if 'timeline' in fname:

        id = fname[:fname.index('_')]

        if flag == 0:
            
            date_col   = csv_to_list(fname, mod='date')[0]
            data_colmn = csv_to_list(fname, mod='date')[1]

            if max(data_colmn) > treshold_val:
                data_colms.append(csv_to_list(fname, mod='date')[1])
                ID.append(id)
                flag = 1
                continue
            else:
                continue

        if flag == 1:
            data_colms.append(csv_to_list(fname))
            ID.append(id)

# transformiere (Transponierung) das Array mit den einzelnen Datenspalten
# um sie alle in einer Zeile zu haben
data_colms = list(np.asarray(data_colms).T)

with open('./merged_data.csv', 'w') as fout:

    i = 0
    for date_time in date_col:
        if i == 0:
            fout.write('date;')
            fout.write(';'.join(ID) + '\n')

        fout.write(date_col[i] + ';' + ';'.join([str(val) for val in data_colms[i]]) + '\n')

        i += 1



