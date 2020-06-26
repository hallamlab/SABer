import logging
import pandas as pd
from os.path import isfile, basename, getsize
from os.path import join as o_join
from subprocess import Popen
from sklearn.preprocessing import normalize
from samsum import commands
import saber.utilities as s_utils
import ray
import sys
from statistics import NormalDist
import multiprocessing
pd.options.mode.chained_assignment = None
from scipy.stats import norm
import numpy as np
import os
from psutil import virtual_memory


def calc_OVL(val):
    m1, m2, std1, std2 = val
    ovl = NormalDist(mu=m1, sigma=std1).overlap(NormalDist(mu=m2, sigma=std2))

    return ovl


@ray.remote
def run_ovl_analysis(s_df, recruit_contigs_df):
    ava_df = pd.merge(recruit_contigs_df, s_df, on='key').drop('key',axis=1)
    ava_df.columns = ['recruit_id', 'r_mu', 'r_sigma', 'query_id', 'q_mu', 'q_sigma']
    ava_filter_df = ava_df.loc[((ava_df['r_sigma'] != 0.0) & (ava_df['q_sigma'] != 0.0))]
    ava_filter_df['ovl'] = [calc_OVL(x) for x in
                  zip(ava_filter_df['r_mu'], ava_filter_df['q_mu'],
                  ava_filter_df['r_sigma'], ava_filter_df['q_sigma'])
                  ]
    ava_final_df = ava_filter_df.loc[ava_filter_df['ovl'] >= 0.99]
    dedup_df = ava_final_df.drop_duplicates(subset='query_id', keep='first')

    return dedup_df


def run_abund_recruiter(subcontig_path, abr_path, mg_sub_file, mg_raw_file_list,
                        minhash_df, covm_per_pass, nthreads
                        ):

    mg_id = mg_sub_file[0]
    mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
    mg_headers = tuple(mg_subcontigs.keys())

    #mg_id, mg_headers = mg_subcontigs[0], mg_subcontigs[1]
    '''
    logging.info('[SABer]: Starting Abundance Recruitment Algorithm\n')
    logging.info('[SABer]: Checking for abundance table for %s\n' % mg_id)
    if isfile(o_join(abr_path, mg_id + '.samsum_merged.tsv')):
        logging.info('[SABer]: Loading  %s abundance table\n' % mg_id)
        mg_ss_df = pd.read_csv(o_join(abr_path, mg_id + '.samsum_merged.tsv'),
                               sep='\t', header=0
                               )
    else:
    '''
    logging.info('[SABer]: Building %s abundance table\n' % mg_id)
    mg_sub_path = o_join(subcontig_path, mg_id + '.subcontigs.fasta')
    # is it indexed?
    index_ext_list = ['amb', 'ann', 'bwt', 'pac', 'sa']
    check_ind_list = ['.'.join([mg_sub_path, x]) for x in index_ext_list]
    if False in (isfile(f) for f in check_ind_list):
        # Use BWA to build an index for metagenome assembly
        logging.info('[SABer]: Creating index with BWA\n')
        bwa_cmd = ['bwa', 'index', mg_sub_path]
        with open(o_join(abr_path, mg_id + '.stdout.txt'), 'w') as stdout_file:
            with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
                run_bwa = Popen(bwa_cmd, stdout=stdout_file,
                                stderr=stderr_file
                                )
                run_bwa.communicate()

    # Process raw metagenomes to calculate abundances
    with open(mg_raw_file_list, 'r') as raw_fa_in:
        raw_data = raw_fa_in.readlines()
    covm_output_list = []
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
        # build bam file
        mg_bam_out = o_join(abr_path, pe_id + '.bam')
        if isfile(mg_bam_out) == False:
            logging.info('[SABer]: Converting SAM to BAM with SamTools\n')
            bam_cmd = ['samtools', 'view', '-S', '-b', '-@', str(nthreads), mg_sam_out]
            with open(mg_bam_out, 'w') as bam_file:
                with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
                    run_bam = Popen(bam_cmd, stdout=bam_file, stderr=stderr_file)
                    run_bam.communicate()
        # sort bam file
        mg_sort_out = o_join(abr_path, pe_id + '.sorted.bam')
        if isfile(mg_sort_out) == False:
            logging.info('[SABer]: Sort BAM with SamTools\n')
            sort_cmd = ['samtools', 'sort', '-@', str(nthreads), mg_bam_out, '-o', mg_sort_out]
            with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
                run_sort = Popen(sort_cmd, stderr=stderr_file)
                run_sort.communicate()
       # run coverm on sorted bam
        mg_covm_out = o_join(abr_path, pe_id + '.metabat.tsv')
        try:
            covm_size = getsize(mg_covm_out)
        except:
            covm_size = -1
        if covm_size <= 0: #isfile(mg_covm_out) == False:
            logging.info('[SABer]: Calculate mean abundance and variance with CoverM\n')
            covm_cmd = ['coverm', 'contig', '-t', str(nthreads), '-b', mg_sort_out, '-m',
                        'metabat']
            with open(mg_covm_out, 'w') as covm_file:
                with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
                    run_covm = Popen(covm_cmd, stdout=covm_file, stderr=stderr_file)
                    run_covm.communicate()
        covm_output_list.append(mg_covm_out)

        '''
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
        '''

    '''
    for mhr_file in mhr_files_list:
        # hackin to get BINID
        if 'evo_' in mhr_file:
            bin_id = '.'.join(mhr_file.split('.', 2)[0:2])
        else:
            mhr_spl = mhr_file.split('.', 1)[0]
            if mhr_spl.find('_'):
                bin_id = '_'.join(mhr_spl.split('_', 2)[0:2])
            else:
                bin_id = mhr_spl
        print(bin_id)

        recruit_df = pd.read_csv(os.path.join(mhr_path, mhr_file), header=None,
                                 names=['sag_id', 'subcontig_id', 'contig_id'],
                                 sep='\t'
                                 )

        sag_id = recruit_df['sag_id'].iloc[0]
    '''
    covm_pass_dfs = []
    for sag_id in set(minhash_df['sag_id']):
        # subset df for sag_id
        sag_mh_pass_df = minhash_df[minhash_df['sag_id'] == sag_id]

        '''
        # subset mapping file using bin_id
        submap_df = mapping_df.loc[mapping_df['BINID'] == bin_id]
        submap_list = list(submap_df['@@SEQUENCEID'])

        files_list = ['metabat_tables/RH_S001__insert_270.metabat.tsv',
                      'metabat_tables/RH_S002__insert_270.metabat.tsv',
                      'metabat_tables/RH_S003__insert_270.metabat.tsv',
                      'metabat_tables/RH_S004__insert_270.metabat.tsv',
                      'metabat_tables/RH_S005__insert_270.metabat.tsv'
                      ]
        '''

        overall_recruit_list = []
        max_mem = int(virtual_memory().total*0.25)
        ray.init(num_cpus=nthreads, memory=max_mem, object_store_memory=max_mem)
        for input_file in covm_output_list:
            print("Starting analysis of {}".format(input_file))
            input_df = pd.read_csv(input_file, header=0, sep='\t')
            input_df.columns = ['contigName', 'contigLeg', 'totalAvgDepth',
                                'AvgDepth', 'variance'
                               ]
            input_df['stdev'] = input_df['variance']**(1/2)

            recruit_contigs_df = input_df[['contigName', 'totalAvgDepth', 'stdev']].loc[
                                            input_df['contigName'].isin(
                                            list(sag_mh_pass_df['subcontig_id']))
                                            ]
            nonrecruit_contigs_df = input_df[['contigName', 'totalAvgDepth', 'stdev']]

            recruit_contigs_df['key'] = 1
            nonrecruit_contigs_df['key'] = 1
            split_nr_dfs = np.array_split(nonrecruit_contigs_df, 100, axis=0)
            r_recruit_contigs_df = ray.put(recruit_contigs_df)
            futures = []
            for i, s_df in enumerate(split_nr_dfs):
                futures.append(run_ovl_analysis.remote(s_df, r_recruit_contigs_df))
                logging.info('\r[SABer]: OVL comparison {0:.0%} complete'.format(
                                                                        i/len(split_nr_dfs)
                                                                        ))
            logging.info('\n')

            ray_results = [r_df for r_df in ray.get(futures)]
            merge_df = pd.concat(ray_results)
            uniq_df = merge_df.drop_duplicates(subset='query_id', keep='first')
            overall_recruit_list.extend(list(uniq_df['query_id']))
            print("Finished")
        ray.shutdown()

        '''
            split_nr_dfs = np.array_split(nonrecruit_contigs_df, 100, axis=0)
            merge_list = []
            num_cores = multiprocessing.cpu_count()
            print("Building multiprocessing pool")
            pool = multiprocessing.Pool(processes=num_cores)
            arg_list = [[i, s_df, recruit_contigs_df] for i, s_df in enumerate(split_nr_dfs)]
            results = pool.imap_unordered(run_ovl_analysis, arg_list)
            print("Executing pool")
            merge_list = []
            for i, o_df in enumerate(results):
                sys.stderr.write('\rdone {0:.0%}'.format(i/len(arg_list)))
                merge_list.append(o_df)
            pool.close()
            pool.join()

            print("\nMerging results")
            merge_df = pd.concat(merge_list)
            uniq_df = merge_df.drop_duplicates(subset='query_id', keep='first')
            overall_recruit_list.extend(list(uniq_df['query_id']))
            print("Finished")
        '''

        uniq_recruit_dict = dict.fromkeys(overall_recruit_list, 0)
        for r in overall_recruit_list:
            uniq_recruit_dict[r] += 1

        final_pass_list = []
        for i in uniq_recruit_dict.items():
            k, v = i
            if (int(v) == int(len(covm_output_list))):
                final_pass_list.append([sag_id, k, k.rsplit('_', 1)[0]])
        final_pass_df = pd.DataFrame(final_pass_list,
                                     columns=['sag_id', 'subcontig_id', 'contig_id']
                                     )
        final_pass_df.to_csv(o_join(abr_path, sag_id + '.abr_recruits.tsv'),
                             header=False, index=False, sep='\t'
                             )
        print("There are {} total subcontigs, {} contigs".format(
              len(final_pass_df['subcontig_id']), len(final_pass_df['contig_id'].unique()))
              )
        covm_pass_dfs.append(final_pass_df)

    covm_df = pd.concat(covm_pass_dfs)

    '''
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
    '''
    # Count # of subcontigs recruited to each SAG via samsum
    covm_cnt_df = covm_df.groupby(['sag_id', 'contig_id']).count().reset_index()
    covm_cnt_df.columns = ['sag_id', 'contig_id', 'subcontig_recruits']
    # Build subcontig count for each MG contig
    mg_contig_list = [x.rsplit('_', 1)[0] for x in mg_headers]
    mg_tot_df = pd.DataFrame(zip(mg_contig_list, mg_headers),
                             columns=['contig_id', 'subcontig_id'])
    mg_tot_cnt_df = mg_tot_df.groupby(['contig_id']).count().reset_index()
    mg_tot_cnt_df.columns = ['contig_id', 'subcontig_total']
    covm_recruit_df = covm_cnt_df.merge(mg_tot_cnt_df, how='left', on='contig_id')
    covm_recruit_df['percent_recruited'] = covm_recruit_df['subcontig_recruits'] / \
                                           covm_recruit_df['subcontig_total']
    covm_recruit_df.sort_values(by='percent_recruited', ascending=False, inplace=True)
    # Only pass contigs that have the magjority of subcontigs recruited (>= 51%)
    covm_recruit_filter_df = covm_recruit_df.loc[covm_recruit_df['percent_recruited'] >=
                                                 float(covm_per_pass)
                                                 ]
    mg_contig_per_max_df = covm_recruit_filter_df.groupby(['contig_id'])[
        'percent_recruited'].max().reset_index()
    mg_contig_per_max_df.columns = ['contig_id', 'percent_max']
    covm_recruit_max_df = covm_recruit_filter_df.merge(mg_contig_per_max_df, how='left',
                                                       on='contig_id')
    # Now pass contigs that have the maximum recruit % of subcontigs
    covm_max_only_df = covm_recruit_max_df.loc[covm_recruit_max_df['percent_recruited'] >=
                                               covm_recruit_max_df['percent_max']
                                               ]
    covm_max_df = covm_df[covm_df['contig_id'].isin(tuple(covm_max_only_df['contig_id']))]

    covm_max_df.to_csv(o_join(abr_path, mg_id + '.abr_trimmed_recruits.tsv'), sep='\t',
                        index=False
                        )

    sys.exit()

    return covm_max_df
