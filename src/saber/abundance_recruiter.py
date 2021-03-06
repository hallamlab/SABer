import logging
import pandas as pd
from os.path import isfile, basename
from os.path import join as o_join
from subprocess import Popen
from sklearn.preprocessing import normalize
from samsum import commands
import saber.utilities as s_utils


def run_abund_recruiter(subcontig_path, abr_path, mg_sub_file, mg_raw_file_list,
                        minhash_df, ss_per_pass, nthreads
                        ):

    mg_id = mg_sub_file[0]
    mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
    mg_headers = tuple(mg_subcontigs.keys())

    #mg_id, mg_headers = mg_subcontigs[0], mg_subcontigs[1]

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
            bwa_cmd = ['bwa', 'index', '-b', '500000000', mg_sub_path] #TODO: how to get install path for executables?
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
        for line in raw_data:
            split_line = line.strip('\n').split('\t')
            if len(split_line) == 2:
                logging.info('[SABer]: Raw reads in FWD and REV file...\n')
                pe1 = split_line[0]
                pe2	= split_line[1]
                mem_cmd = ['bwa', 'mem', '-t', str(nthreads), '-p',
                           o_join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1, pe2
                           ] #TODO: add support for specifying number of threads
            else: # if the fastq is interleaved
                logging.info('[SABer]: Raw reads in interleaved file...\n')
                pe1 = split_line[0]
                mem_cmd = ['bwa', 'mem', '-t', str(nthreads), '-p',
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
                        run_mem = Popen(mem_cmd, stdout=sam_file, stderr=stderr_file)
                        run_mem.communicate()

            logging.info('[SABer]: Calculating TPM with samsum for %s\n' % pe_id)
            mg_input = o_join(subcontig_path, mg_id + '.subcontigs.fasta')
            sam_input = o_join(abr_path, pe_id + '.sam')
            # samsum API
            ref_seq_abunds = commands.ref_sequence_abundances(aln_file=sam_input,
                                                              seq_file=mg_input, multireads=True
                                                              )
            ss_output_list.append(ref_seq_abunds)

        logging.info('[SABer]: Merging results for all samsum output\n')
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
        mg_ss_df.to_csv(o_join(abr_path, mg_id + '.samsum_merged.tsv'), sep='\t', index=False)

    # extract TPM and pivot for MG
    mg_ss_trim_df = mg_ss_df[['subcontig_id', 'sample_index', 'tpm']].dropna(how='any')
    mg_ss_piv_df = pd.pivot_table(mg_ss_trim_df, values='tpm', index='subcontig_id',
                                  columns='sample_index')
    normed_ss_df = pd.DataFrame(normalize(mg_ss_piv_df.values),
                                  columns=mg_ss_piv_df.columns,
                                  index=mg_ss_piv_df.index
                                  )
    normed_ss_df.to_csv(o_join(abr_path, mg_id + '.samsum_normmed.tsv'),
                          sep='\t'
                          )
    # get MinHash "passed" mg ss
    ss_pass_list = []
    for sag_id in set(minhash_df['sag_id']):
        logging.info('[SABer]: Calulating/Loading abundance stats for %s\n' % sag_id)
        if isfile(o_join(abr_path, sag_id + '.abr_recruits.tsv')):
            with open(o_join(abr_path, sag_id + '.abr_recruits.tsv'), 'r') as abr_in:
                pass_list = tuple([x.rstrip('\n').split('\t') for x in abr_in.readlines()])
        else:
            sag_mh_pass_df = minhash_df[minhash_df['sag_id'] == sag_id]
            mh_cntg_pass_list = set(sag_mh_pass_df['subcontig_id'])
            mg_ss_pass_df = mg_ss_piv_df[
                mg_ss_piv_df.index.isin(mh_cntg_pass_list)
            ]
            mg_ss_test_df = mg_ss_piv_df[
                ~mg_ss_piv_df.index.isin(mh_cntg_pass_list)
            ]

            mg_ss_pass_stat_df = mg_ss_pass_df.mean().reset_index()
            mg_ss_pass_stat_df.columns = ['sample_id', 'mean']
            mg_ss_pass_stat_df['std'] = tuple(mg_ss_pass_df.std())
            mg_ss_pass_stat_df['var'] = tuple(mg_ss_pass_df.var())
            mg_ss_pass_stat_df['skew'] = tuple(mg_ss_pass_df.skew())
            mg_ss_pass_stat_df['kurt'] = tuple(mg_ss_pass_df.kurt())
            mg_ss_pass_stat_df['IQ_25'] = tuple(mg_ss_pass_df.quantile(0.25))
            mg_ss_pass_stat_df['IQ_75'] = tuple(mg_ss_pass_df.quantile(0.75))
            mg_ss_pass_stat_df['IQ_10'] = tuple(mg_ss_pass_df.quantile(0.10))
            mg_ss_pass_stat_df['IQ_90'] = tuple(mg_ss_pass_df.quantile(0.90))
            mg_ss_pass_stat_df['IQ_05'] = tuple(mg_ss_pass_df.quantile(0.05))
            mg_ss_pass_stat_df['IQ_95'] = tuple(mg_ss_pass_df.quantile(0.95))
            mg_ss_pass_stat_df['IQ_01'] = tuple(mg_ss_pass_df.quantile(0.01))
            mg_ss_pass_stat_df['IQ_99'] = tuple(mg_ss_pass_df.quantile(0.99))
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
            join_ss_recruits = set(tuple(iqr_pass_df.index) + tuple(mh_cntg_pass_list))
            for md_nm in join_ss_recruits:
                pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
            with open(o_join(abr_path, sag_id + '.abr_recruits.tsv'), 'w') as abr_out:
                abr_out.write('\n'.join(['\t'.join(x) for x in pass_list]))
        logging.info('[SABer]: Recruited %s subcontigs to %s\n' % (len(pass_list), sag_id))
        ss_pass_list.extend(tuple(pass_list))

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
                                                 float(ss_per_pass)
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
    ss_max_df = ss_df[ss_df['contig_id'].isin(tuple(ss_max_only_df['contig_id']))]

    ss_max_df.to_csv(o_join(abr_path, mg_id + '.abr_trimmed_recruits.tsv'), sep='\t',
                        index=False
                        )

    return ss_max_df
