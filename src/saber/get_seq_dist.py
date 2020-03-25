import saber.utilities as s_utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import math

def roundup(x, step):
    return int(math.ceil(x / float(step))) * step


step = 100

fasta_dir = sys.argv[1]
fasta_list = [os.path.join(fasta_dir, x) for x in os.listdir(fasta_dir) if (('.fasta' in x)
              | ('.fna' in x))
              ]
sag_seq_list = []
max_len = step
for i, fasta_file in enumerate(fasta_list):
    print(fasta_file)
    sag_id = i
    fasta_dat = s_utils.get_seqs(fasta_file)
    for j, seq_rec in enumerate(fasta_dat):
        header, seq = seq_rec
        sag_seq_list.append([sag_id, j, len(seq)])
        if len(seq) > max_len:
            max_len = roundup(len(seq), step)
print(len(fasta_list))
print(max_len)
sag_seq_df = pd.DataFrame(sag_seq_list, columns=['sag_id', 'contig_id', 'seq_len'])
interval_range = pd.interval_range(start=0, freq=step, end=max_len)
sag_seq_df['n-tiles'] = pd.cut(sag_seq_df['seq_len'], bins=interval_range).astype(str)
sag_list = list(sag_seq_df['sag_id'].unique())
print(sag_seq_df.head())
cnt_bins_df = sag_seq_df.groupby(['sag_id','n-tiles'])['contig_id'].count().reset_index()
cnt_bins_df.columns = ['sag_id', 'n-tiles', 'count']
cnt_bins_df['bins'] = [int(x.split(',')[0].strip('(')) for x in cnt_bins_df['n-tiles']]
cnt_bins_df.sort_values(by=['bins'], inplace=True)
print(cnt_bins_df.head())
piv_df = cnt_bins_df.pivot(index='bins', columns='sag_id', values='count')
piv_df.fillna(0, inplace=True)
piv_df.reset_index(inplace=True)
print(piv_df.head())
piv_df.to_csv('SAG_Catelog_bins.tsv', sep='\t', index=False)
sys.exit()




n_tiles_list = list(sag_seq_df['n-tiles'].unique())
tmp_list = [0 for r in range(len(sag_list))]
bin_dict = {x:tmp_list for x in n_tiles_list}
print(len(tmp_list))
print(len(sag_list))
for s in sag_list:
    sub_sag_df = sag_seq_df.loc[sag_seq_df['sag_id'] == s]
    val_cnts = zip(sub_sag_df['n-tiles'].value_counts().index.tolist(), sub_sag_df['n-tiles'].value_counts().tolist())
    for vc in val_cnts:
        k = vc[0]
        c = vc[1]
        print(k, c)
        bin_dict[k][sag_list.index(s)] = c
col_list = ['bin'] + sag_list
print(col_list)
for k in list(bin_dict.keys()):
    print(k, bin_dict[k])
    sys.exit()
bin_list = [[x[0]] + x[1] for x in bin_dict.items()]

bin_df = pd.DataFrame(bin_list, columns=col_list)
bin_df.sort_values('bin', inplace=True)
print(bin_df.head())
bin_df.to_csv('SAG_Catelog_bins.tsv', sep='\t', index=False)






sys.exit()
bin_dict = {k:[] for k in range(0, max_len, step)}
sag_id_list = []
for sag_id in sag_seq_dict.keys():
    print(sag_id)
    sag_id_list.append(sag_id)
    x = []
    fasta_dat = sag_seq_dict[sag_id]
    for seq_rec in fasta_dat:
        header, seq = seq_rec
        x.append(len(seq))
    n, bins, patches = plt.hist(x, bins=int(max_len/step), range=[0, max_len])
    for b, v in zip(bins, n):
        bin_dict[b].append(v)

col_list = ['bin'] + sag_id_list
bin_list = [[x[0]] + x[1] for x in bin_dict.items()]
bin_df = pd.DataFrame(bin_list, columns=col_list)
bin_df.sort_values('bin', inplace=True)
print(bin_df.head())
bin_df.to_csv('SAG_Catelog_bins.tsv', sep='\t', index=False)

#ave_bin_dict = {x:(sum(bin_dict[x])/len(bin_dict[x])) for x in bin_dict.keys()}
#ave_bin_df = pd.DataFrame(ave_bin_dict.items(), columns=['bin', 'ave_cnt'])
#ave_bin_df.sort_values('bin', inplace=True)
#print(ave_bin_df.head())
#ave_bin_df.to_csv('SI_SAG_bins_average.tsv', sep='\t', index=False)
#ax = sns.barplot(x='bin', y='ave_cnt', data=bin_df)
#plt.savefig('bar.pdf')



