import saber.utilities as s_utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

fasta_dir = sys.argv[1]
fasta_list = [os.path.join(fasta_dir, x) for x in os.listdir(fasta_dir) if '.fasta' in x]
bin_dict = {k:[] for k in range(0, 350000, 1000)}
for fasta_file in fasta_list:
	print(fasta_file)
	x = []
	fasta_dat = s_utils.get_seqs(fasta_file)
	for seq_rec in fasta_dat:
		header, seq = seq_rec
		x.append(len(seq))
	n, bins, patches = plt.hist(x, bins=350, range=[0, 350000]) 
	for b, v in zip(bins, n):
		bin_dict[b].append(v)
ave_bin_dict = {x:(sum(bin_dict[x])/len(bin_dict[x])) for x in bin_dict.keys()}
bin_df = pd.DataFrame(ave_bin_dict.items(), columns=['bin', 'ave_cnt'])
bin_df.sort_values('bin', inplace=True)
print(bin_df.head())
bin_df.to_csv('SI_SAG_bins.tsv', sep='\t', index=False)
#ax = sns.barplot(x='bin', y='ave_cnt', data=bin_df)
#plt.savefig('bar.pdf')



