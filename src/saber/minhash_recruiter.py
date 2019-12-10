#####################################################################################
###########################                               ###########################
########################### MinHash Recruitment Algorithm ###########################
###########################                               ###########################
#####################################################################################
print('[SABer]: Starting MinHash Recruitment Algorithm')

# Calculate/Load MinHash Signatures with SourMash for MG subseqs
if isfile(join(sig_path, mg_id + '.metaG.sig')): 
	print('[SABer]: Loading %s Signatures' % mg_id)
	mg_sig_list = sourmash.signature.load_signatures(join(sig_path, mg_id + \
														'.metaG.sig')
														)
else:
	print('[SABer]: Building Signatures for %s' % mg_id)
	mg_sig_list = []
	for mg_head, seq in mg_sub_tup:
		up_seq = seq.upper()
		mg_minhash = sourmash.MinHash(n=0, ksize=51, scaled=100)
		mg_minhash.add_sequence(up_seq, force=True)
		mg_sig = sourmash.SourmashSignature(mg_minhash, name=mg_head)
		mg_sig_list.append(mg_sig)
	with open(join(sig_path, mg_id + '.metaG.sig'), 'w') as mg_out:
		sourmash.signature.save_signatures(mg_sig_list,	fp=mg_out)

# Load comparisons OR Compare SAG sigs to MG sigs to find containment
print('[SABer]: Comparing Signatures of SAGs to MetaG contigs')
minhash_pass_list = []
for sag_id, sag_sub_tup in sag_contigs_dict.items():
	if isfile(join(mhr_path, sag_id + '.mhr_recruits.tsv')):
		print('[SABer]: Loading  %s and MetaG signature recruit list' % sag_id)
		with open(join(mhr_path, sag_id + '.mhr_recruits.tsv'), 'r') as mhr_in:
			pass_list = [x.rstrip('\n').split('\t') for x in mhr_in.readlines()]
	else:
		# Calculate\Load MinHash Signatures with SourMash for SAG subseqs
		if isfile(join(sig_path, sag_id + '.SAG.sig')):
			print('[SABer]: Loading Signature for %s' % sag_id)
			sag_sig = sourmash.signature.load_one_signature(join(sig_path,
																sag_id + '.SAG.sig')
																)
		else:
			print('[SABer]: Building Signature for %s' % sag_id)
			sag_minhash = sourmash.MinHash(n=0, ksize=51, scaled=100)
			for sag_head, sag_subseq in sag_sub_tup:
				sag_upseq = sag_subseq.upper()
				sag_minhash.add_sequence(sag_upseq, force=True)
			sag_sig = sourmash.SourmashSignature(sag_minhash, name=sag_id)
			with open(join(sig_path, sag_id + '.SAG.sig'), 'w') as sags_out:
				sourmash.signature.save_signatures([sag_sig], fp=sags_out)
		print('[SABer]: Comparing  %s and MetaG signatures' % sag_id)
		pass_list = []
		mg_sig_list = list(mg_sig_list)
		for j, mg_sig in enumerate(mg_sig_list):
			jacc_sim = mg_sig.contained_by(sag_sig)
			mg_nm = mg_sig.name()
			if jacc_sim >= 0.95:
				pass_list.append([sag_id, mg_nm, mg_nm.rsplit('_', 1)[0]])

		with open(join(mhr_path, sag_id + '.mhr_recruits.tsv'), 'w') as mhr_out:
			mhr_out.write('\n'.join(['\t'.join(x) for x in pass_list]))
	minhash_pass_list.extend(pass_list)
	print('[SABer]: Recruited subcontigs to %s' % sag_id)

minhash_df = pd.DataFrame(minhash_pass_list, columns=['sag_id', 'subcontig_id',
														'contig_id'
														])
