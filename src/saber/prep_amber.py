import pandas as pd
import sys
from pathlib import Path

input_file = sys.argv[1]
input_df = pd.read_csv(input_file, header=0, sep='\t')

amber_map_file = sys.argv[2]
amber_map_df = pd.read_csv(amber_map_file, header=0, sep='\t')

merge_df = pd.merge(input_df, amber_map_df, on='sag_id')

amber_df = merge_df[['contig_id', 'AMBER_ID']]
amber_df.columns = ['@@SEQUENCEID', 'BINID']

sag_tax_map = '/home/rmclaughlin/Ryan/CAMI_I_HIGH/genome_taxa_info.tsv'
sag_taxmap_df = pd.read_csv(sag_tax_map, sep='\t', header=0)
sag_taxmap_df['sp_taxid'] = [int(x) for x in sag_taxmap_df['@@TAXID']]
sag_taxmap_df['sp_name'] = [x.split('|')[-2] for x in sag_taxmap_df['TAXPATHSN']]
sag_taxmap_df['TAXID_substr'] = sag_taxmap_df['@@TAXID']
sag_taxmap_df['@@TAXID'] = [str(x).split('.')[0] for x in sag_taxmap_df['@@TAXID']]
sag_taxmap_df.columns = ['_TAXID_', '_RANK_', '_TAXPATH_', '_TAXPATHSN_', '_PERCENTAGE_',
								'BINID', '_CAMI_OTU_', '_sp_taxid_', '_sp_name_',
								'_TAXID_substr_'
								]

amber_tax_df = pd.merge(amber_df, sag_taxmap_df, on='BINID')
amber_tax_df = amber_tax_df[['@@SEQUENCEID', 'BINID', '_TAXID_', '_RANK_', '_TAXPATH_',
								'_TAXPATHSN_', '_PERCENTAGE_', '_CAMI_OTU_',
								'_sp_taxid_', '_sp_name_','_TAXID_substr_'
								]]

with open(input_file.rsplit('.', 1)[0] + '.AMBER.binning', 'w') as a_out:
	a_out.write('@Version:0.9.1\n@SampleID:CAMI_HIGH\n\n')							
amber_tax_df.to_csv(input_file.rsplit('.', 1)[0] + '.AMBER.binning', index=False, sep='\t',
						mode='a')

'''
for bin_id in set(amber_tax_df['BINID']):
	sub_amber_tax_df = amber_tax_df.loc[amber_tax_df['BINID'] == bin_id]
	sv_dir = input_file.rsplit('.', 1)[0].replace('.', '_')
	Path(sv_dir).mkdir(parents=True, exist_ok=True)
	bin_file = sv_dir + '/' + bin_id + '.' + input_file.rsplit('.', 1)[0] + '.binning'
	with open(bin_file, 'w') as a_out:
		a_out.write('@Version:0.9.1\n@SampleID:CAMI_HIGH\n\n')							
	sub_amber_tax_df.to_csv(bin_file, index=False, sep='\t', mode='a')
'''
