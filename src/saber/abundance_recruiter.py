import logging
import pandas as pd
from os.path import isfile, basename
from os.path import join as o_join
from subprocess import Popen
from sklearn.preprocessing import normalize
from samsum import commands, file_parsers
import samsum.alignment_utils as ss_aln_utils

import sys

def run_abund_recruiter(subcontig_path, abr_path, mg_subcontigs, mg_raw_file_list,
                        minhash_df, ss_per_pass=0.51
                        ):
    mg_id, mg_headers, mg_subs = mg_subcontigs
    logging.info('[SABer]: Starting Abundance Recruitment Algorithm\n')
    logging.info('[SABer]: Checking for abundance table for %s\n' % mg_id)
    if isfile(o_join(abr_path, mg_id + '.samsum_merged.tsv')):
        logging.info('[SABer]: Loading  %s abundance table\n' % mg_id)
        mg_ss_df = pd.read_csv(o_join(abr_path, mg_id + '.samsum_merged.tsv'),
                               sep='\t', header=0
                               )
    else:
        logging.info('[SABer]: Building %s abundance table\n' % mg_id)
        mg_sub_path = o_join(subcontig_path, mg_id + '.subcontigs.fasta')
        # is is indexed?
        index_ext_list = ['amb', 'ann', 'bwt', 'pac', 'sa']
        check_ind_list = ['.'.join([mg_sub_path, x]) for x in index_ext_list]
        if False in (isfile(f) for f in check_ind_list):
            # Use BWA to build an index for metagenome assembly
            logging.info('[SABer]: Creating index with BWA\n')
            bwa_cmd = ['bwa', 'index', mg_sub_path] #TODO: how to get install path for executables?
            with open(o_join(abr_path, mg_id + '.stdout.txt'), 'w') as stdout_file:
                with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
                    run_bwa = Popen(bwa_cmd, stdout=stdout_file,
                                    stderr=stderr_file
                                    )
                    run_bwa.communicate()

        # Process raw metagenomes to calculate abundances
        with open(mg_raw_file_list, 'r') as raw_fa_in:
            raw_data = raw_fa_in.readlines()
        ss_output_list = []
        sample_count = 0
        for line in raw_data:
            split_line = line.strip('\n').split('\t')
            if len(split_line) == 2:
                logging.info('[SABer]: Raw reads in FWD and REV file...\n')
                pe1 = split_line[0]
                pe2	= split_line[1]
                mem_cmd = ['bwa', 'mem', '-t', '2', '-p',
                           o_join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1, pe2
                           ] #TODO: how to get install path for executables?
            else: # if the fastq is interleaved
                logging.info('[SABer]: Raw reads in interleaved file...\n')
                pe1 = split_line[0]
                mem_cmd = ['bwa', 'mem', '-t', '2', '-p',
                           o_join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1
                           ] #TODO: how to get install path for executables?
            pe_basename = basename(pe1)
            pe_id = pe_basename.split('.')[0]
            # BWA sam file exists?
            mg_sam_out = o_join(abr_path, pe_id + '.sam')
            if isfile(mg_sam_out) == False:
                logging.info('[SABer]: Running BWA mem on %s\n' % pe_id)
                with open(mg_sam_out, 'w') as sam_file:
                    with open(o_join(abr_path, pe_id + '.stderr.txt'), 'w') as stderr_file:
                        run_mem = Popen(mem_cmd, stdout=sam_file,
                                        stderr=stderr_file
                                        )
                        run_mem.communicate()

            logging.info('[SABer]: Calculating TPM with samsum for %s\n' % pe_id) # TODO: re-enable the SS API
            mg_input = o_join(subcontig_path, mg_id + '.subcontigs.fasta')
            sam_input = o_join(abr_path, pe_id + '.sam')
            mg_ss_out = sam_input.rsplit('.', 1)[0] + '.ss.csv'
            ss_cmd = ['samsum', 'stats', '-f', mg_input, '-a', sam_input, '--multireads',
                      '-o', mg_ss_out
                      ]
            with open(o_join(abr_path, pe_id + '.ss_log.txt'), 'w') as ss_log:
                run_ss = Popen(ss_cmd, stdout=ss_log, stderr=ss_log)
                run_ss.communicate()
            ss_df = pd.read_csv(mg_ss_out, header=0, sep=',')
            ss_df['rso_sample_index'] = sample_count
            ss_output_list.append(ss_df)
            sample_count += 1

            '''
            # implement samsum API
            ref_seq_abunds = commands.ref_sequence_abundances(aln_file=sam_input,
                                                              seq_file=mg_input, multireads=True
                                                              )
            '''
        logging.info('[SABer]: Merging results for all samsum output\n')
        '''
        # Merge API output for each raw sample file
        refseq_header_list = ss_output_list[0].keys()
        refseq_merge_list = []
        for refseq_header in refseq_header_list:
            for i, refseq_dict in enumerate(ss_output_list):
                refseq_obj = refseq_dict[refseq_header]
                rso_name = refseq_obj.name
                rso_length = refseq_obj.length
                rso_reads_mapped = refseq_obj.reads_mapped
                rso_weight_total = refseq_obj.weight_total
                rso_fpkm = refseq_obj.fpkm
                rso_tpm = refseq_obj.tpm
                rso_sample_index = i
                refseq_merge_list.append([rso_name, rso_sample_index, rso_length,
                                          rso_reads_mapped, rso_weight_total, rso_fpkm,
                                          rso_tpm
                                          ])
        mg_ss_df = pd.DataFrame(refseq_merge_list, columns=['subcontig_id', 'sample_index',
                                                              'length', 'reads_mapped',
                                                              'weight_total', 'fpkm', 'tpm'
                                                              ])
        '''
        mg_ss_df = pd.concat(ss_output_list)
        # re-calculate TPM and intermediates
        mg_ss_df['new_RPK'] = mg_ss_df['Fragments']/(mg_ss_df['contig_len']/1000)
        samp2rpkmsum = {}
        for sample in set(mg_ss_df['rso_sample_index']):
            sub_ss_df = mg_ss_df.loc[mg_ss_df['rso_sample_index'] == sample]
            sum_RPKM = sub_ss_df['new_RPK'].sum(axis=0)
            samp2rpkmsum[sample] = sum_RPKM
        mg_ss_df['per_mil_scaler'] = [samp2rpkmsum[x]/1e6 for x in mg_ss_df['rso_sample_index']]
        mg_ss_df['new_TPM'] = mg_ss_df['new_RPK']/mg_ss_df['per_mil_scaler']
        mg_ss_df.to_csv(o_join(abr_path, mg_id + '.samsum_merged.tsv'), sep='\t', index=False)
    # extract TPM and pivot for MG
    mg_ss_trim_df = mg_ss_df[['RefSequence', 'rso_sample_index', 'new_TPM']]
    mg_ss_trim_df.columns = ['subcontig_id', 'sample_index', 'tpm']
    mg_ss_trim_df.dropna(how='any', inplace=True)
    # TODO: pivot sample_index column to create TPM matrix
    mg_ss_piv_df = pd.pivot_table(mg_ss_trim_df, values='tpm', index='subcontig_id',
                                  columns='sample_index')
    # Normalize data # TODO: is this needed now? Probs not :)
    normed_ss_df = pd.DataFrame(normalize(mg_ss_piv_df.values),
                                  columns=mg_ss_piv_df.columns,
                                  index=mg_ss_piv_df.index
                                  )
    normed_ss_df.to_csv(o_join(abr_path, mg_id + '.samsum_normmed.tsv'),
                          sep='\t'
                          )
    # get MinHash "passed" mg sss
    ss_pass_list = []
    for sag_id in set(minhash_df['sag_id']):
        logging.info('[SABer]: Calulating/Loading abundance stats for %s\n' % sag_id)
        if isfile(o_join(abr_path, sag_id + '.abr_recruits.tsv')):
            with open(o_join(abr_path, sag_id + '.abr_recruits.tsv'), 'r') as abr_in:
                pass_list = [x.rstrip('\n').split('\t') for x in abr_in.readlines()]
        else:
            sag_mh_pass_df = minhash_df[minhash_df['sag_id'] == sag_id]
            mh_cntg_pass_list = set(sag_mh_pass_df['subcontig_id'])
            mg_ss_pass_df = normed_ss_df[
                normed_ss_df.index.isin(mh_cntg_pass_list)
            ]
            mg_ss_test_df = normed_ss_df[
                ~normed_ss_df.index.isin(mh_cntg_pass_list)
            ]

            mg_ss_pass_stat_df = mg_ss_pass_df.mean().reset_index()
            mg_ss_pass_stat_df.columns = ['sample_id', 'mean']
            mg_ss_pass_stat_df['std'] = list(mg_ss_pass_df.std())
            mg_ss_pass_stat_df['var'] = list(mg_ss_pass_df.var())
            mg_ss_pass_stat_df['skew'] = list(mg_ss_pass_df.skew())
            mg_ss_pass_stat_df['kurt'] = list(mg_ss_pass_df.kurt())
            mg_ss_pass_stat_df['IQ_25'] = list(mg_ss_pass_df.quantile(0.25))
            mg_ss_pass_stat_df['IQ_75'] = list(mg_ss_pass_df.quantile(0.75))
            mg_ss_pass_stat_df['IQ_10'] = list(mg_ss_pass_df.quantile(0.10))
            mg_ss_pass_stat_df['IQ_90'] = list(mg_ss_pass_df.quantile(0.90))
            mg_ss_pass_stat_df['IQ_05'] = list(mg_ss_pass_df.quantile(0.05))
            mg_ss_pass_stat_df['IQ_95'] = list(mg_ss_pass_df.quantile(0.95))
            mg_ss_pass_stat_df['IQ_01'] = list(mg_ss_pass_df.quantile(0.01))
            mg_ss_pass_stat_df['IQ_99'] = list(mg_ss_pass_df.quantile(0.99))
            mg_ss_pass_stat_df['IQR'] = mg_ss_pass_stat_df['IQ_75'] - \
                                          mg_ss_pass_stat_df['IQ_25']
            # calc Tukey Fences
            mg_ss_pass_stat_df['upper_bound'] = mg_ss_pass_stat_df['IQ_75'] + \
                                                  (1.5 * mg_ss_pass_stat_df['IQR'])
            mg_ss_pass_stat_df['lower_bound'] = mg_ss_pass_stat_df['IQ_25'] - \
                                                  (1.5 * mg_ss_pass_stat_df['IQR'])

            mg_ss_pass_stat_df.to_csv(o_join(abr_path, sag_id + '.passed_ss_stats.tsv'),
                                        sep='\t'
                                        )

            # Use passed MG from MHR to recruit more seqs,
            iqr_pass_df = mg_ss_test_df.copy()
            for i, col_nm in enumerate(mg_ss_test_df.columns):
                pass_stats = mg_ss_pass_stat_df.iloc[[i]]
                pass_max = pass_stats['upper_bound'].values[0]
                pass_min = pass_stats['lower_bound'].values[0]
                iqr_pass_df = iqr_pass_df.loc[(iqr_pass_df[col_nm] >= pass_min) &
                                              (iqr_pass_df[col_nm] <= pass_max)
                                              ]

            pass_list = []
            join_ss_recruits = set(list(iqr_pass_df.index) + list(mh_cntg_pass_list))
            for md_nm in join_ss_recruits:
                pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])

            logging.info('[SABer]: Recruited %s subcontigs to %s\n' % (len(pass_list), sag_id))
            with open(o_join(abr_path, sag_id + '.abr_recruits.tsv'), 'w') as abr_out:
                abr_out.write('\n'.join(['\t'.join(x) for x in pass_list]))

        ss_pass_list.extend(pass_list)

    ss_df = pd.DataFrame(ss_pass_list, columns=['sag_id', 'subcontig_id',
                                                    'contig_id'
                                                    ])
    # Count # of subcontigs recruited to each SAG via samsum
    ss_cnt_df = ss_df.groupby(['sag_id', 'contig_id']).count().reset_index()
    ss_cnt_df.columns = ['sag_id', 'contig_id', 'subcontig_recruits']
    # Build subcontig count for each MG contig
    mg_contig_list = [x.rsplit('_', 1)[0] for x in mg_headers]
    mg_tot_df = pd.DataFrame(zip(mg_contig_list, mg_headers),
                             columns=['contig_id', 'subcontig_id'])
    mg_tot_cnt_df = mg_tot_df.groupby(['contig_id']).count().reset_index()
    mg_tot_cnt_df.columns = ['contig_id', 'subcontig_total']
    ss_recruit_df = ss_cnt_df.merge(mg_tot_cnt_df, how='left', on='contig_id')
    ss_recruit_df['percent_recruited'] = ss_recruit_df['subcontig_recruits'] / \
                                           ss_recruit_df['subcontig_total']
    ss_recruit_df.sort_values(by='percent_recruited', ascending=False, inplace=True)
    # Only pass contigs that have the magjority of subcontigs recruited (>= 51%)
    ss_recruit_filter_df = ss_recruit_df.loc[ss_recruit_df['percent_recruited'] >=
                                                 ss_per_pass
                                                 ]
    mg_contig_per_max_df = ss_recruit_filter_df.groupby(['contig_id'])[
        'percent_recruited'].max().reset_index()
    mg_contig_per_max_df.columns = ['contig_id', 'percent_max']
    ss_recruit_max_df = ss_recruit_filter_df.merge(mg_contig_per_max_df, how='left',
                                                       on='contig_id')
    # Now pass contigs that have the maximum recruit % of subcontigs
    ss_max_only_df = ss_recruit_max_df.loc[ss_recruit_max_df['percent_recruited'] >=
                                               ss_recruit_max_df['percent_max']
                                               ]
    ss_max_df = ss_df[ss_df['contig_id'].isin(list(ss_max_only_df['contig_id']))]

    return ss_max_df