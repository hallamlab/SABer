import pandas as pd
pd.set_option('display.max_columns', None)
import os
import glob
import numpy as np
import sys



barnap_dir = sys.argv[1]
data_list = []
empty_list = []
bar_check = [os.path.basename(x).rsplit('.', 2)[0] for x in
				glob.glob(os.path.join(barnap_dir, "*.rRNA.fasta"))
				]
blast_dict = {os.path.basename(x).rsplit('.', 1)[0]: x for x in 
				glob.glob(os.path.join(barnap_dir, "*.blastout"))
				}
for bar_id in bar_check:
	print(bar_id)
	if bar_id in blast_dict.keys():
		bar_file = blast_dict[bar_id]
		with open(bar_file, 'r') as bar_in:
			data = bar_in.readlines()
			if len(data) != 0:
				for line in data:
					if line != '':
						split_line = line.strip('\n').split('\t')
						split_line.insert(0,bar_id)
						data_list.append(split_line)
	else:
		empty_list.append(bar_id)

blast_header = ['Bin Id', 'qaccver', 'saccver', 'pident',
				'length', 'mismatch', 'gapopen', 'qstart',
				'qend', 'sstart', 'send', 'evalue', 'bitscore'
			    ]
blast_df = pd.DataFrame(data_list, columns=blast_header)

blast_df['SSU'] = [x.split('_', 1)[0] for x in
					  blast_df['qaccver'].copy()
					  ]
blast_df['sSSU'] = [x.split('_', 1)[0] for x in
					  blast_df['saccver'].copy()
					  ]

blast_df = blast_df.query('SSU == sSSU')

col_list = ['Bin Id']
blast_df['pident'] = pd.to_numeric(blast_df['pident'].copy())
blast_df['bitscore'] = pd.to_numeric(blast_df['bitscore'].copy())
blast_df.to_csv(os.path.join(barnap_dir, 'barrnap_blast_raw.tsv'), sep='\t', index=False)

min_id_df = blast_df.groupby(col_list)[['pident', 'bitscore']].min().reset_index()

min_id_df['pass_BARRNAP'] = [True if ((x[0] >= 97) & (x[1] >= 100)) else False
							 for x in zip(min_id_df['pident'], min_id_df['bitscore'])
							 ]

empty_df = pd.DataFrame(empty_list, columns=['Bin Id'])
empty_df['pident'] = -1
empty_df['bitscore'] = -1
empty_df['pass_BARRNAP'] = True	  

cat_df = pd.concat([min_id_df, empty_df])
print(cat_df.head())

cat_df.to_csv(os.path.join(barnap_dir, 'barrnap_blast_pass.tsv'), sep='\t', index=False)

bar_gffs = [x for x in glob.glob(os.path.join(barnap_dir, "*.rRNA.gff"))]
gff_list = []
for gff in bar_gffs:
	try:
		gff_df = pd.read_csv(gff, sep='\t', header=None, comment='#')
		gff_df.columns = ['seqid', 'source', 'type', 'start', 'end',
						  'score', 'strand', 'phase', 'attributes'
						  ]
		gff_df['genome_id'] = os.path.basename(gff).split('.rRNA.')[0]
		gff_df['name'] = [x.split(';', 1)[0].split('=')[1] for x in gff_df['attributes']]
		gff_df['product'] = [x.split(';', 2)[1].split('=')[1] for x in gff_df['attributes']]
		gff_df['prod_mod'] = [int(x.split(';', 3)[2].split(
							  'aligned only ')[1].split(' percent of')[0])/100
							  if ';note' in x else 1.0 for x in gff_df['attributes']
							  ]
		gff_list.append(gff_df)
	except:
		print(gff, 'is likely empty...')
gff_cat_df = pd.concat(gff_list)
gff_cat_df.to_csv(os.path.join(barnap_dir, 'barrnap_gff_table.tsv'), sep='\t', index=False)

gff_grp_df = gff_cat_df.groupby(['genome_id', 'name'])['prod_mod'].sum().reset_index()
gff_grp_df.rename(columns = {"prod_mod": "subunit_count"}, inplace = True)
gff_grp_df.to_csv(os.path.join(barnap_dir, 'barrnap_subunit_counts.tsv'), sep='\t', index=False)






