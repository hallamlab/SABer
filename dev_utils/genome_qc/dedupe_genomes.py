import pandas as pd
pd.set_option('display.max_columns', None)
import math
import sys
import numpy as np
import os
import glob


working_dir = sys.argv[1]
comp = int(sys.argv[2])
cont = int(sys.argv[3])
qsmin = int(sys.argv[4])
qscore_file = os.path.join(working_dir, "qscore/qscore_mqhp.tsv")  # MQHP genomes list
barrnap_file = os.path.join(working_dir, "barrnap/barrnap_blast_pass.tsv")  # barrnap blast results
gtdbtk_files = glob.glob(os.path.join(working_dir, "gtdbtk/classify/gtdbtk.*.summary.tsv"))  # bac and arc gtdbtk results
qscore_df = pd.read_csv(qscore_file, header=0, sep='\t')
barrnap_df = pd.read_csv(barrnap_file, header=0, sep='\t')
cat_list = []
for g in gtdbtk_files:
	g_df = pd.read_csv(g, header=0, sep='\t')
	g_df['Bin Id'] = [x for x in g_df['user_genome']]
	cat_list.append(g_df)
if len(cat_list) > 1:
	gtdbtk_df = pd.concat(cat_list)
else:
	gtdbtk_df = cat_list[0]
merge_list = ['Bin Id']
qscore_df = qscore_df.merge(barrnap_df, on=merge_list, how="left")
qscore_df = qscore_df.merge(gtdbtk_df, on=merge_list, how="left")
print(qscore_df.head())

qscore_df.to_csv(os.path.join(working_dir, "dedupe/dedupe_all.tsv"), index=False, sep='\t')

sorted_df = qscore_df.sort_values(by=['qscore', 'N50', 'sum_len'],
				ascending=[False, False, False]
				)
keep_cols = ['Genome_Id', 'Bin Id', 'classification', 'Completeness', 'Contamination', 'Strain heterogeneity',
		'num_seqs', 'sum_len', 'min_len', 'avg_len', 'max_len', 'N50', 'qscore', 'pass_BARRNAP']
mq_df = sorted_df[keep_cols].query("Completeness >= @comp & Contamination <= @cont & qscore >= @qsmin & pass_BARRNAP == True").drop_duplicates('Genome_Id')
mq_df.to_csv(os.path.join(working_dir, 'dedupe/dedupe_mqhp.tsv'), index=False, sep='\t')

concat_top_list = []
for gen_id in list(mq_df['Genome_Id']):
	mq_comp = mq_df.query('Genome_Id == @gen_id')['Completeness'].values[0]
	mq_cont = mq_df.query('Genome_Id == @gen_id')['Contamination'].values[0]
	gen_tot_df = sorted_df.query("Genome_Id == @gen_id & Completeness == @mq_comp & Contamination == @mq_cont & pass_BARRNAP == True")
	concat_top_list.append(gen_tot_df)
tot_df = pd.concat(concat_top_list)
tot_df.to_csv(os.path.join(working_dir, 'dedupe/total_mqhp.tsv'), index=False, sep='\t')
