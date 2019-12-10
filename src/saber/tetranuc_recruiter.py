#####################################################################################
##################                                                 ##################
################## Tetranucleotide Frequency Recruitment Algorithm ##################
##################                                                 ##################
#####################################################################################
# TODO: Think about using Minimum Description Length (MDL) instead of AIC/BIC
#		[Normalized Maximum Likelihood or Fish Information Approximation]
# Build/Load tetramers for SAGs and MG subset by ara recruits
if isfile(join(tra_path, mg_id + '.tetras.tsv')):
	print('[SABer]: Loading tetramer Hz matrix for %s' % mg_id)
	mg_tetra_df = pd.read_csv(join(tra_path, mg_id + '.tetras.tsv'),
								sep='\t',index_col=0, header=0
								)
else:
	print('[SABer]: Calculating tetramer Hz matrix for %s' % mg_id)
	mg_tetra_df = pd.DataFrame.from_dict(tetra_cnt(mg_subs))
	mg_tetra_df['contig_id'] = mg_headers
	mg_tetra_df.set_index('contig_id', inplace=True)
	mg_tetra_df.to_csv(join(tra_path, mg_id + '.tetras.tsv'),
						sep='\t'
						)

gmm_pass_list = []
for sag_id, sag_sub_tup in sag_subcontigs_dict.items():
	sag_headers = sag_sub_tup[0]
	sag_subs = sag_sub_tup[1]

	if isfile(join(tra_path, sag_id + '.tra_recruits.tsv')):
		print('[SABer]: Loading  %s tetramer Hz recruit list' % sag_id)
		with open(join(tra_path, sag_id + '.tra_recruits.tsv'), 'r') as tra_in:
			pass_list = [x.rstrip('\n').split('\t') for x in tra_in.readlines()]
	else:
		if isfile(join(tra_path, sag_id + '.tetras.tsv')):
			print('[SABer]: Loading tetramer Hz matrix for %s' % sag_id)
			sag_tetra_df = pd.read_csv(join(tra_path, sag_id + '.tetras.tsv'),
										sep='\t', index_col=0, header=0)
		else:
			print('[SABer]: Calculating tetramer Hz matrix for %s' % sag_id)
			sag_tetra_df = pd.DataFrame.from_dict(tetra_cnt(sag_subs))
			sag_tetra_df['contig_id'] = sag_headers
			sag_tetra_df.set_index('contig_id', inplace=True)
			sag_tetra_df.to_csv(join(tra_path, sag_id + '.tetras.tsv'), sep='\t')
		
		# Concat SAGs amd MG for GMM
		mg_rpkm_contig_list = list(rpkm_max_df.loc[rpkm_max_df['sag_id'] == sag_id
												]['subcontig_id'].values
												)
		# get list of all RPKM recruits
		mg_rpkm_pass_index = [x for x in mg_tetra_df.index
						if x in mg_rpkm_contig_list
						]
		# get list of subset of RPKM complete failures
		# subset fail df to size of sag tetra df
		n_val = len(list(sag_tetra_df.index))
		rpkm_sub_fail_df = rpkm_max_fail_df.sample(n=n_val, random_state=42)

		mg_rpkm_contig_fail_list = list(rpkm_sub_fail_df.loc[rpkm_sub_fail_df['sag_id'] == sag_id
												]['subcontig_id'].values
												)
		mg_rpkm_fail_index = [x for x in mg_tetra_df.index
						if x in mg_rpkm_contig_fail_list
						]

		mg_tetra_filter_df = mg_tetra_df.loc[mg_tetra_df.index.isin(mg_rpkm_pass_index)]
		# Build Fail tetra df
		mg_tetra_fail_df = mg_tetra_df.loc[mg_tetra_df.index.isin(mg_rpkm_fail_index)]


		concat_tetra_df = pd.concat([sag_tetra_df, mg_tetra_filter_df, mg_tetra_fail_df])
		normed_tetra_df = pd.DataFrame(normalize(concat_tetra_df.values),
										columns=concat_tetra_df.columns,
										index=concat_tetra_df.index
										)
		sag_normed_tetra_df = normed_tetra_df[
								normed_tetra_df.index.isin(sag_tetra_df.index)
								]
		mg_normed_tetra_df = normed_tetra_df.loc[
								normed_tetra_df.index.isin(mg_tetra_filter_df.index)
								]

		# UMAP for Dimension reduction of tetras
		sag_features = sag_normed_tetra_df.values
		sag_targets = sag_normed_tetra_df.index.values
		mg_features = mg_normed_tetra_df.values
		mg_targets = mg_normed_tetra_df.index.values
		normed_features = normed_tetra_df.values
		normed_targets = normed_tetra_df.index.values
		print('[SABer]: Dimension reduction of tetras with UMAP')
		umap_trans = umap.UMAP(n_neighbors=2, min_dist=0.0,
						n_components=num_components, metric='manhattan',
						random_state=42
						).fit_transform(normed_features)

		pc_col_names = ['pc' + str(x) for x in range(1, num_components + 1)]
		umap_df = pd.DataFrame(umap_trans, columns=pc_col_names, index=normed_targets)

		sag_umap_df = umap_df.loc[umap_df.index.isin(sag_tetra_df.index)]
		mg_umap_df = umap_df.loc[umap_df.index.isin(mg_tetra_filter_df.index)]
		mg_fail_umap_df = umap_df.loc[umap_df.index.isin(mg_tetra_fail_df.index)]
		train_df = pd.concat([sag_umap_df, mg_fail_umap_df])
		train_vals = [1 if x in list(sag_tetra_df.index) else 0 for x in train_df.index]
		n_components = np.arange(1, 100, 1)
		models = [GMM(n, random_state=42)
			  for n in n_components]
		bics = []
		aics = []
		for i, model in enumerate(models):
			n_comp = n_components[i]
			try:
				bic = model.fit(train_df.values,
								train_vals).bic(train_df.values
								)
				bics.append(bic)
			except:
				print('[WARNING]: BIC failed with %s components' % n_comp)
			try:
				aic = model.fit(train_df.values,
								train_vals).aic(train_df.values
								)
				aics.append(aic)
			except:
				print('[WARNING]: AIC failed with %s components' % n_comp)

		min_bic_comp = n_components[bics.index(min(bics))]
		min_aic_comp = n_components[aics.index(min(aics))]
		print('[SABer]: Min AIC/BIC at %s/%s, respectively' % 
				(min_aic_comp, min_bic_comp)
				)
		print('[SABer]: Using AIC as guide for GMM components')
		print('[SABer]: Training GMM on SAG tetras')
		gmm = GMM(n_components=min_aic_comp, random_state=42
						).fit(train_df.values, train_vals
						)
		print('[SABer]: GMM Converged: ', gmm.converged_)
		try:
			train_scores = gmm.score_samples(train_df.values)
			sag_scores = gmm.score_samples(sag_umap_df.values)
			sag_scores_df = pd.DataFrame(data=sag_scores, index=sag_targets)
			sag_score_min = min(sag_scores_df.values)[0]
			sag_score_max = max(sag_scores_df.values)[0]
			mg_scores = gmm.score_samples(mg_umap_df.values)
			mg_scores_df = pd.DataFrame(data=mg_scores, index=mg_targets)
			gmm_pass_df = mg_scores_df.loc[(mg_scores_df[0] >= sag_score_min) &
											(mg_scores_df[0] <= sag_score_max)
											]
			# And is has to be from the RPKM pass list
			gmm_pass_df = gmm_pass_df.loc[gmm_pass_df.index.isin(mg_rpkm_pass_index)]

			pass_list = []
			for md_nm in gmm_pass_df.index.values:
				pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
		except:
			print('[SABer]: Warning: No recruits found...')
			pass_list = []

		
		'''
		##################################
		# build scatterplot to viz the GMM
		sag_xy_df = sag_umap_df.iloc[:,0:2].copy()
		mg_xy_df = mg_umap_df.iloc[:,0:2].copy()
		sag_xy_df['isSAG'] = 'SAG'
		mg_xy_df['isSAG'] = ['Tetra-Recruit' if x in list(gmm_pass_df.index.values)
								else 'MG' for x in mg_xy_df.index.values
								]
		recruits_xy_df = mg_xy_df[mg_xy_df['isSAG'] == 'Tetra-Recruit']
		mg_xy_df = mg_xy_df[mg_xy_df['isSAG'] == 'MG']
		xy_df = pd.concat([mg_xy_df, sag_xy_df, recruits_xy_df])
		xy_df.to_csv(join(tra_path, sag_id + '.GMM_plot.tsv'), sep='\t')
		sv_plot = join(tra_path, sag_id + '.GMM_plot.png')
		ax = sns.scatterplot(x='pc1', y='pc2', data=xy_df, hue='isSAG',
								alpha=0.4, edgecolor='none')
		plt.gca().set_aspect('equal', 'datalim')
		plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.savefig(sv_plot, bbox_inches="tight")
		plt.clf()
		##################################
		'''

		print('[SABer]: Recruited %s subcontigs to %s' % (len(pass_list), sag_id))
		with open(join(tra_path, sag_id + '.tra_recruits.tsv'), 'w') as tra_out:
			tra_out.write('\n'.join(['\t'.join(x) for x in pass_list]))
	gmm_pass_list.extend(pass_list)
gmm_df = pd.DataFrame(gmm_pass_list, columns=['sag_id', 'subcontig_id', 'contig_id'])
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

