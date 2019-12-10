#####################################################################################
##########################                                 ##########################
########################## Abundance Recruitment Algorithm ##########################
##########################                                 ##########################
#####################################################################################
# NOTE: This is built to accept output from 'join_rpkm_out.py' script
# TODO: Add RPKM cmd call to run within this script
print('[SABer]: Starting Abundance Recruitment Algorithm')

print('[SABer]: Checking for RPKM values table for %s' % mg_id)
if isfile(join(ara_path, mg_id + '.rpkm.tsv')):
	print('[SABer]: Loading  %s RPKM table' % mg_id)
	mg_rpkm_df = pd.read_csv(join(ara_path, mg_id + '.rpkm.tsv'), sep='\t', header=0)
else:
	print('[SABer]: Building %s RPKM table' % mg_id)
	# Use BWA to build an index for metagenome assembly
	print('[SABer]: Creating index with BWA')
	bwa_cmd = ['bwa', 'index',
				join(subcontig_path, mg_id + '.subcontigs.fasta')
				]
	with open(join(ara_path, mg_id + '.stdout.txt'), 'w') as stdout_file:
		with open(join(ara_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
			run_bwa = Popen(bwa_cmd, stdout=stdout_file,
							stderr=stderr_file
							)
			run_bwa.communicate()

	# Process raw metagenomes to calculate RPKMs
	with open(mg_raw_file_list, 'r') as raw_fa_in:
		raw_data = raw_fa_in.readlines()
	rpkm_output_list = []
	for line in raw_data:
		split_line = line.strip('\n').split('\t')
		if len(split_line) == 2:
			pe1 = split_line[0]
			pe2	= split_line[1]
			mem_cmd = ['bwa', 'mem', '-t', '2',
				join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1, pe2
				]
		else: # if the fastq is interleaved
			pe1 = split_line[0]
			mem_cmd = ['bwa', 'mem', '-t', '2',
				join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1
				]
		pe_basename = basename(pe1)
		pe_id = pe_basename.split('.')[0]
		print('[SABer]: Running BWA mem on %s' % pe_id)
		with open(join(ara_path, pe_id + '.sam'), 'w') as sam_file:
			with open(join(ara_path, pe_id + '.stderr.txt'), 'w') as stderr_file:
				run_mem = Popen(mem_cmd, stdout=sam_file,
								stderr=stderr_file
								)
				run_mem.communicate()

		print('[SABer]: Calculating RPKM for %s' % pe_id)
		rpkm_cmd = ['rpkm',
						'-c', join(subcontig_path, mg_id + '.subcontigs.fasta'),
						'-a', join(ara_path, pe_id + '.sam'),
						'-o', join(ara_path, pe_id + '.rpkm.csv')
						]
		with open(join(ara_path, pe_id + '.rpkm_stdout.log'), 'w') as stdlog_file:
			with open(join(ara_path, pe_id + '.rpkm_stderr.log'), 'w') as stderr_file:
				run_rpkm = Popen(rpkm_cmd, stdout=stdlog_file,
									stderr=stderr_file
									)
				run_rpkm.communicate()

		rpkm_output_list.append(join(ara_path, pe_id + '.rpkm.csv'))

	print('[SABer]: Merging RPKM results for all raw data')
	merge_cmd = ['python', 'join_rpkm_out.py',
					','.join(rpkm_output_list), join(ara_path, mg_id + '.rpkm.tsv')
					]
	with open(join(ara_path, mg_id + '.merge_stdout.log'), 'w') as stdmerge_file:
		with open(join(ara_path, mg_id + '.merge_stderr.log'), 'w') as stderr_file:
			run_merge = Popen(merge_cmd, stdout=stdmerge_file,
								stderr=stderr_file
								)
			run_merge.communicate()

	mg_rpkm_df = pd.read_csv(join(ara_path, mg_id + '.rpkm.tsv'), sep='\t', header=0)

mg_rpkm_col_list = ['Sequence_name']
for col in mg_rpkm_df.columns:
	if 'RPKM' in col:
		mg_rpkm_col_list.append(col)
mg_rpkm_trim_df = mg_rpkm_df[mg_rpkm_col_list]
mg_rpkm_trim_df = mg_rpkm_trim_df.loc[mg_rpkm_trim_df['Sequence_name']
										!= 'UNMAPPED'
										]
mg_rpkm_trim_df.set_index('Sequence_name', inplace=True)
mg_rpkm_trim_df.to_csv(join(ara_path, mg_id + '.cleaned_rpkm.tsv'),
						sep='\t'
						)
# Normalize data
normed_rpkm_df = pd.DataFrame(normalize(mg_rpkm_trim_df.values),
							columns=mg_rpkm_trim_df.columns,
							index=mg_rpkm_trim_df.index
							)
normed_rpkm_df.to_csv(join(ara_path, mg_id + '.normmed_rpkm.tsv'),
						sep='\t'
						)
# get MinHash "passed" mg rpkms
rpkm_pass_list = []
rpkm_fail_list = []	
for sag_id in set(minhash_df['sag_id']):
	print('[SABer]: Calulating/Loading RPKM stats for %s' % sag_id)
	if isfile(join(ara_path, sag_id + '.ara_recruits.tsv')):
		with open(join(ara_path, sag_id + '.ara_recruits.tsv'), 'r') as ara_in:
			pass_list = [x.rstrip('\n').split('\t') for x in ara_in.readlines()]
		with open(join(ara_path, sag_id + '.ara_failed.tsv'), 'r') as ara_in:
			fail_list = [x.rstrip('\n').split('\t') for x in ara_in.readlines()]
	else:
		sag_mh_pass_df = minhash_df[minhash_df['sag_id'] == sag_id]
		mh_cntg_pass_list = set(sag_mh_pass_df['subcontig_id'])
		mg_rpkm_pass_df = normed_rpkm_df[
									normed_rpkm_df.index.isin(mh_cntg_pass_list)
									]
		mg_rpkm_test_df = normed_rpkm_df[
									~normed_rpkm_df.index.isin(mh_cntg_pass_list)
									]
		
		mg_rpkm_pass_stat_df = mg_rpkm_pass_df.mean().reset_index()
		mg_rpkm_pass_stat_df.columns = ['sample_id', 'mean']
		mg_rpkm_pass_stat_df['std'] = list(mg_rpkm_pass_df.std())
		mg_rpkm_pass_stat_df['var'] = list(mg_rpkm_pass_df.var())
		mg_rpkm_pass_stat_df['skew'] = list(mg_rpkm_pass_df.skew())
		mg_rpkm_pass_stat_df['kurt'] = list(mg_rpkm_pass_df.kurt())
		mg_rpkm_pass_stat_df['IQ_25'] = list(mg_rpkm_pass_df.quantile(0.25))
		mg_rpkm_pass_stat_df['IQ_75'] = list(mg_rpkm_pass_df.quantile(0.75))
		mg_rpkm_pass_stat_df['IQ_10'] = list(mg_rpkm_pass_df.quantile(0.10))
		mg_rpkm_pass_stat_df['IQ_90'] = list(mg_rpkm_pass_df.quantile(0.90))
		mg_rpkm_pass_stat_df['IQ_05'] = list(mg_rpkm_pass_df.quantile(0.05))
		mg_rpkm_pass_stat_df['IQ_95'] = list(mg_rpkm_pass_df.quantile(0.95))
		mg_rpkm_pass_stat_df['IQ_01'] = list(mg_rpkm_pass_df.quantile(0.01))
		mg_rpkm_pass_stat_df['IQ_99'] = list(mg_rpkm_pass_df.quantile(0.99))
		mg_rpkm_pass_stat_df['IQR'] = mg_rpkm_pass_stat_df['IQ_75'] - \
										mg_rpkm_pass_stat_df['IQ_25']
		# calc Tukey Fences
		mg_rpkm_pass_stat_df['upper_bound'] = mg_rpkm_pass_stat_df['IQ_75'] + \
												(1.5 * mg_rpkm_pass_stat_df['IQR'])
		mg_rpkm_pass_stat_df['lower_bound'] = mg_rpkm_pass_stat_df['IQ_25'] - \
												(1.5 * mg_rpkm_pass_stat_df['IQR'])

		mg_rpkm_pass_stat_df.to_csv(join(ara_path, sag_id + '.passed_rpkm_stats.tsv'),
									sep='\t'
									)

		# Use passed MG from MHR to recruit more seqs,
		# also build fail df for GMM training
		iqr_pass_df = mg_rpkm_test_df.copy()
		iqr_fail_df = mg_rpkm_test_df.copy()
		for i, col_nm in enumerate(mg_rpkm_test_df.columns):
			pass_stats = mg_rpkm_pass_stat_df.iloc[[i]]
			pass_max = pass_stats['upper_bound'].values[0]
			pass_min = pass_stats['lower_bound'].values[0]
			iqr_pass_df = iqr_pass_df.loc[(iqr_pass_df[col_nm] >= pass_min) &
											(iqr_pass_df[col_nm] <= pass_max)
											]
			iqr_fail_df = iqr_fail_df.loc[(iqr_fail_df[col_nm] < pass_min) |
											(iqr_fail_df[col_nm] > pass_max)
											]

		pass_list = []
		join_rpkm_recruits = set(list(iqr_pass_df.index) + list(mh_cntg_pass_list))
		for md_nm in join_rpkm_recruits:
			pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])

		fail_list = []
		join_rpkm_failures = set(list(iqr_fail_df.index))
		for md_nm in join_rpkm_failures:
			fail_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])

		print('[SABer]: Recruited %s subcontigs to %s' % (len(pass_list), sag_id))
		print('[SABer]: %s subcontigs failed RPKM filter for %s' % (len(fail_list), sag_id))
		with open(join(ara_path, sag_id + '.ara_recruits.tsv'), 'w') as ara_out:
			ara_out.write('\n'.join(['\t'.join(x) for x in pass_list]))
		with open(join(ara_path, sag_id + '.ara_failed.tsv'), 'w') as ara_out:
			ara_out.write('\n'.join(['\t'.join(x) for x in fail_list]))

	rpkm_pass_list.extend(pass_list)
	rpkm_fail_list.extend(fail_list)

rpkm_df = pd.DataFrame(rpkm_pass_list, columns=['sag_id', 'subcontig_id',
												'contig_id'
												])
# Count # of subcontigs recruited to each SAG via rpkm
rpkm_cnt_df = rpkm_df.groupby(['sag_id', 'contig_id']).count().reset_index()
rpkm_cnt_df.columns = ['sag_id', 'contig_id', 'subcontig_recruits']
# Build subcontig count for each MG contig
mg_contig_list = [x.rsplit('_', 1)[0] for x in mg_headers]
mg_tot_df = pd.DataFrame(zip(mg_contig_list, mg_headers),
								columns=['contig_id', 'subcontig_id'])
mg_tot_cnt_df = mg_tot_df.groupby(['contig_id']).count().reset_index()
mg_tot_cnt_df.columns = ['contig_id', 'subcontig_total']
rpkm_recruit_df = rpkm_cnt_df.merge(mg_tot_cnt_df, how='left', on='contig_id')
rpkm_recruit_df['percent_recruited'] = rpkm_recruit_df['subcontig_recruits'] / \
										rpkm_recruit_df['subcontig_total']
rpkm_recruit_df.sort_values(by='percent_recruited', ascending=False, inplace=True)
# Only pass contigs that have the magjority of subcontigs recruited (>= 51%)
rpkm_recruit_filter_df = rpkm_recruit_df.loc[rpkm_recruit_df['percent_recruited'] >=
												rpkm_per_pass
												]
mg_contig_per_max_df = rpkm_recruit_filter_df.groupby(['contig_id'])[
										'percent_recruited'].max().reset_index()
mg_contig_per_max_df.columns = ['contig_id', 'percent_max']
rpkm_recruit_max_df = rpkm_recruit_filter_df.merge(mg_contig_per_max_df, how='left',
												on='contig_id')
# Now pass contigs that have the maximum recruit % of subcontigs
rpkm_max_only_df = rpkm_recruit_max_df.loc[rpkm_recruit_max_df['percent_recruited'] >=
											rpkm_recruit_max_df['percent_max']
											]
rpkm_max_df = rpkm_df[rpkm_df['contig_id'].isin(list(rpkm_max_only_df['contig_id']))]


# Now the same for the fails, but only keep 100% failures
rpkm_fail_df = pd.DataFrame(rpkm_fail_list, columns=['sag_id', 'subcontig_id',
												'contig_id'
												])
# Count # of subcontigs recruited to each SAG via rpkm
rpkm_fail_cnt_df = rpkm_fail_df.groupby(['sag_id', 'contig_id']).count().reset_index()
rpkm_fail_cnt_df.columns = ['sag_id', 'contig_id', 'subcontig_recruits']
# Build subcontig count for each MG contig
mg_contig_list = [x.rsplit('_', 1)[0] for x in mg_headers]
mg_tot_df = pd.DataFrame(zip(mg_contig_list, mg_headers),
								columns=['contig_id', 'subcontig_id'])
mg_tot_cnt_df = mg_tot_df.groupby(['contig_id']).count().reset_index()
mg_tot_cnt_df.columns = ['contig_id', 'subcontig_total']
rpkm_fail_recruit_df = rpkm_fail_cnt_df.merge(mg_tot_cnt_df, how='left', on='contig_id')
rpkm_fail_recruit_df['percent_recruited'] = rpkm_fail_recruit_df['subcontig_recruits'] / \
										rpkm_fail_recruit_df['subcontig_total']
rpkm_fail_recruit_df.sort_values(by='percent_recruited', ascending=False, inplace=True)
rpkm_fail_recruit_filter_df = rpkm_fail_recruit_df.loc[rpkm_fail_recruit_df['percent_recruited'] >=
												1.0
												]
mg_contig_fail_per_max_df = rpkm_fail_recruit_filter_df.groupby(['contig_id'])[
										'percent_recruited'].max().reset_index()
mg_contig_fail_per_max_df.columns = ['contig_id', 'percent_max']
rpkm_recruit_fail_max_df = rpkm_fail_recruit_filter_df.merge(mg_contig_fail_per_max_df, how='left',
												on='contig_id')
# Now pass contigs that have the maximum recruit % of subcontigs
rpkm_max_fail_only_df = rpkm_recruit_fail_max_df.loc[rpkm_recruit_fail_max_df['percent_recruited'] >=
											rpkm_recruit_fail_max_df['percent_max']
											]
rpkm_max_fail_df = rpkm_fail_df[rpkm_fail_df['contig_id'].isin(list(rpkm_max_fail_only_df['contig_id']))]

#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

