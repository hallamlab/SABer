import logging
import pandas as pd
from os.path import isfile, basename, getsize
from os.path import join as o_join
from subprocess import Popen
from sklearn.preprocessing import normalize
from samsum import commands
import saber.utilities as s_utils
import sys
from statistics import NormalDist
import multiprocessing
pd.options.mode.chained_assignment = None
from scipy.stats import norm
import numpy as np
import os
from psutil import virtual_memory
from tqdm import tqdm
import scipy.stats
import itertools
import swifter
import time
import argparse
from sklearn import svm


def calc_OVL(m1, m2, std1, std2):

    ovl = NormalDist(mu=m1, sigma=std1).overlap(NormalDist(mu=m2, sigma=std2))

    return ovl


def run_ovl_analysis(p):
    recruit_df, query_df = p
    ava_df = pd.merge(recruit_df, query_df, on='key').drop('key',axis=1)
    ava_df.columns = ['recruit_id', 'r_mu', 'r_sigma', 'query_id', 'q_mu', 'q_sigma']
    ava_df['mu_diff'] = (ava_df['r_mu'] - ava_df['q_mu']).abs()
    ava_df = ava_df.loc[(((ava_df['r_sigma']/4) >= ava_df['mu_diff'])
                        & ((ava_df['q_sigma']/4) >= ava_df['mu_diff'])
                        )]
    ovl_list = []
    zip_list = list(zip(ava_df['r_mu'], ava_df['q_mu'],
                        ava_df['r_sigma'], ava_df['q_sigma']))
    for x in zip_list:
        ovl_list.append(calc_OVL(x[0], x[1], x[2], x[3]))
    ava_df['ovl'] = ovl_list
    ava_df = ava_df.loc[ava_df['ovl'] >= 0.98]
    ovl_contig_tup = tuple(set(ava_df['query_id']))

    return ovl_contig_tup


def run_svm_analysis(sag_df, mg_df, sag_id):
    # fit OCSVM
    clf = svm.OneClassSVM()
    clf.fit(sag_df.values)
    mg_pred = clf.predict(mg_df.values)
    mg_pred_df = pd.DataFrame(data=mg_pred, index=mg_df.index.values)
    svm_pass_df = mg_pred_df.loc[mg_pred_df[0] != -1]
    svm_pass_list = []
    for md_nm in svm_pass_df.index.values:
        svm_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])

    return svm_pass_list



def run_abund_recruiter(subcontig_path, abr_path, mg_sub_file, mg_raw_file_list,
                        minhash_df, covm_per_pass, nthreads
                        ):

    mg_id = mg_sub_file[0]
    mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
    mg_headers = tuple(mg_subcontigs.keys())

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
    sorted_bam_list = []
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
        sorted_bam_list.append(mg_sort_out)

    # run coverm on sorted bam
    mg_covm_out = o_join(abr_path, mg_id + '.metabat.tsv')
    try: # if file exists but is empty
        covm_size = getsize(mg_covm_out)
    except: # if file doesn't exist
        covm_size = -1
    if covm_size <= 0:
        logging.info('[SABer]: Calculate mean abundance and variance with CoverM\n')
        covm_cmd = ['coverm', 'contig', '-t', str(nthreads), '-m', 'metabat', '-b'
                    ]
        covm_cmd.extend(sorted_bam_list)
        with open(mg_covm_out, 'w') as covm_file:
            with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
                run_covm = Popen(covm_cmd, stdout=covm_file, stderr=stderr_file)
                run_covm.communicate()

    covm_pass_dfs = []
    for sag_id in tqdm(set(minhash_df['sag_id'])):
        if isfile(o_join(abr_path, sag_id + '.abr_recruits.tsv')):
            logging.info('[SABer]: Loading Abundance Recruits for  %s\n' % sag_id)
            final_pass_df = pd.read_csv(o_join(abr_path, sag_id + '.abr_recruits.tsv'),
                                        header=None,
                                        names=['sag_id', 'subcontig_id', 'contig_id'],
                                        sep='\t'
                                        )
            covm_pass_dfs.append(final_pass_df)
        else:
            # subset df for sag_id
            mh_jacc_list = list(set(minhash_df['contig_id'].loc[
                                            (minhash_df['sag_id'] == sag_id) &
                                            (minhash_df['jacc_sim'] >= 0.99)
                                            ]))
            if len(mh_jacc_list) != 0:
                sag_mh_pass_df = minhash_df.loc[minhash_df['contig_id'].isin(mh_jacc_list)]
                overall_recruit_list = []
                logging.info("Starting one-class SVM analysis\n")
                mg_covm_df = pd.read_csv(mg_covm_out, header=0, sep='\t')
                recruit_contigs_df = mg_covm_df.loc[mg_covm_df['contigName'].isin(
                                                list(sag_mh_pass_df['subcontig_id']))
                                                ]
                nonrecruit_filter_df = mg_covm_df.loc[~mg_covm_df['contigName'].isin(
                                                        recruit_contigs_df['contigName'])
                                                        ]
                recruit_contigs_df.drop(columns=['contigLen', 'totalAvgDepth'], inplace=True)
                nonrecruit_filter_df.drop(columns=['contigLen', 'totalAvgDepth'], inplace=True)
                recruit_contigs_df.set_index('contigName', inplace=True)
                nonrecruit_filter_df.set_index('contigName', inplace=True)

                keep_cols = [x for x in recruit_contigs_df.columns if x.rsplit('.', 1)[1] != 'bam-var'] # recruit_contigs_df.columns
                recruit_bam_df = recruit_contigs_df[keep_cols]
                nonrecruit_bam_df = nonrecruit_filter_df[keep_cols]

                final_pass_list = run_svm_analysis(recruit_contigs_df, nonrecruit_filter_df, sag_id)

                '''
                logging.info("Starting OV coefficient analysis\n")
                for input_file in tqdm(covm_output_list):
                    input_df = pd.read_csv(input_file, header=0, sep='\t')
                    input_df.columns = ['contigName', 'contigLeg', 'totalAvgDepth',
                                        'AvgDepth', 'variance'
                                       ]
                    input_df['stdev'] = input_df['variance']**(1/2)
                    filter_df = input_df.loc[input_df['stdev'] != 0.0]
                    filter_df['upper'] = filter_df['totalAvgDepth'] + filter_df['stdev']
                    filter_df['lower'] = filter_df['totalAvgDepth'] - filter_df['stdev']
                    filter_df['key'] = 1

                    recruit_contigs_df = filter_df.loc[filter_df['contigName'].isin(
                                                    list(sag_mh_pass_df['subcontig_id']))
                                                    ]

                    r_max_sd1 = recruit_contigs_df['upper'].max()
                    r_min_sd1 = recruit_contigs_df['lower'].min()

                    nonrecruit_filter_df = filter_df.loc[
                                                ((filter_df['totalAvgDepth'] >= r_min_sd1)
                                                & (filter_df['totalAvgDepth'] <= r_max_sd1)
                                                & ~filter_df['contigName'].isin(
                                                    recruit_contigs_df['contigName']
                                                    ))]
                    recruit_contigs_df = recruit_contigs_df[['contigName', 'totalAvgDepth', 'stdev', 'key']]
                    nonrecruit_filter_df = nonrecruit_filter_df[['contigName', 'totalAvgDepth', 'stdev', 'key']]

                    split_nr_dfs = np.array_split(nonrecruit_filter_df, nthreads*10, axis=0)
                    pool = multiprocessing.Pool(processes=nthreads)
                    arg_list = [[recruit_contigs_df, s_df] for i, s_df in enumerate(split_nr_dfs)]
                    results = pool.imap_unordered(run_ovl_analysis, arg_list)
                    merge_list = []
                    for i, o_list in enumerate(results):
                        sys.stderr.write('\rdone {0:.0%}'.format(i/len(arg_list)))
                        merge_list.extend(o_list)
                    set_list = list(set(merge_list))
                    overall_recruit_list.extend(set_list)
                    pool.close()
                    pool.join()


                uniq_recruit_dict = dict.fromkeys(overall_recruit_list, 0)
                for r in overall_recruit_list:
                    uniq_recruit_dict[r] += 1

                final_pass_list = []
                for i in uniq_recruit_dict.items():
                    k, v = i
                    if (int(v) == int(len(covm_output_list))):
                        final_pass_list.append([sag_id, k, k.rsplit('_', 1)[0]])
                '''
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


    return covm_max_df



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='uses metabat normalized abundance to recruit metaG reads to SAGs')
    parser.add_argument(
        '--abr_path', help='path to abundance output directory',
        required=True
        )
    parser.add_argument(
        '--sub_path',
        help='path to SAG subcontigs file(s)', required=True
        )
    parser.add_argument(
        '--mg_sub_file',
        help='path to metagenome subcontigs file', required=True
        )
    parser.add_argument(
        '--raw_fastqs',
        help='path to raw metagenomes, comma separated list', required=True
        )
    parser.add_argument(
        '--minh_df',
        help='path to output dataframe from abundance recruiter', required=True
        )
    parser.add_argument(
        '--per_pass',
        help='pass percentage of subcontigs to pass complete contig [0.70]', required=True,
        default='0.70'
        )
    parser.add_argument(
        '--threads',
        help='number of threads to use [1]', required=True,
        default='1'
        )
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Prints a more verbose runtime log"
                        )
    args = parser.parse_args()
    # set args
    abr_path = args.abr_path
    subcontig_path = args.sub_path
    mg_sub_file = args.mg_sub_file
    mg_raw_file_list = args.raw_fastqs.split(',')
    minhash_recruit_file = args.minh_df
    per_pass = float(args.per_pass)
    nthreads = int(args.threads)

    s_log.prep_logging("abund_log.txt", args.verbose)
    mg_id = basename(mg_sub_file).rsplit('.', 2)[0]
    minhash_recruit_df = pd.read_csv(minhash_recruit_file, header=0, sep='\t')
    logging.info('[SABer]: Starting Tetranucleotide Recruitment Step\n')

    run_abund_recruiter(subcontig_path, abr_path, [mg_id, mg_sub_file],
                        minhash_recruit_df, per_pass, nthreads)

