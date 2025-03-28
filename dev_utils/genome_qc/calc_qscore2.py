import pandas as pd
pd.set_option('display.max_columns', None)
import math
import sys
import numpy as np
import os
import glob


def calc_qscore(compl, contam, strhet, n50, size):
	A = 1
	B = 5
	C = 1
	D = 0.5
	E = 0
	qscore = \
		(A * compl) - \
		(B * contam) + \
		(C * (contam * (strhet / 100))) + \
		(D * math.log(n50)) + \
		(E * math.log(size))
	return qscore


working_dir = sys.argv[1]
comp = int(sys.argv[2])
cont = int(sys.argv[3])
qscr = int(sys.argv[4])

checkm_file = os.path.join(working_dir, "checkm/checkm_output.20000.tsv")  # checkm output tsv
seqkit_file = os.path.join(working_dir, "seqkit/seqkit_stats.tsv")  # seqkit output tsv
checkm_df = pd.read_csv(checkm_file, header=0, sep='\t')
seqkit_df = pd.read_csv(seqkit_file, header=0, sep='\t')
seqkit_df['Bin Id'] = [x.rsplit('/', 1)[1].rsplit('.', 1)[0] for x in seqkit_df['file']]
merge_list = ['Bin Id']
qscore_df = checkm_df.merge(seqkit_df, on=merge_list, how="left")
print(qscore_df.head())
qscore_df['qscore'] = qscore_df.apply(lambda row: calc_qscore(row['Completeness'],
								row['Contamination'],
								row['Strain heterogeneity'],
								row['N50'],
								row['sum_len']
								), axis=1
						)

qscore_df.to_csv(os.path.join(working_dir, "qscore/qscore_all.tsv"), index=False, sep='\t')

sorted_df = qscore_df.sort_values(by=['qscore', 'N50', 'sum_len'],
				ascending=[False, False, False]
				)
keep_cols = ['Genome_Id', 'Bin Id', 'Completeness', 'Contamination', 'Strain heterogeneity',
		'num_seqs', 'sum_len', 'min_len', 'avg_len', 'max_len', 'N50', 'qscore']
#print(set(sorted_df['pass_GUNC']))
sorted_df['Genome_Id'] = [x.rsplit('.', 1)[0] for x in sorted_df['Bin Id']]
print(sorted_df.head())
mq_df = sorted_df[keep_cols].query("Completeness >= @comp & Contamination <= @cont & qscore >= @qscr")
#mq_df.to_csv(os.path.join(working_dir, 'qscore/qscore_mqhp.tsv'), index=False, sep='\t')
