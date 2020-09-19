import logging
import pandas as pd
import os
from os.path import isfile, basename, getsize
from os.path import join as o_join
from subprocess import Popen
import saber.utilities as s_utils
import numpy as np
from tqdm import tqdm
import argparse
from sklearn import svm
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None


def runAbundRecruiter(subcontig_path, abr_path, mg_sub_file, mg_raw_file_list,
                        minhash_df, covm_per_pass, nthreads
                        ):

    mg_id = mg_sub_file[0]
    mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
    mg_headers = tuple(mg_subcontigs.keys())
    logging.info('[SABer]: Building %s abundance table\n' % mg_id)
    mg_sub_path = o_join(subcontig_path, mg_id + '.subcontigs.fasta')
    # Process raw metagenomes to calculate abundances
    mg_covm_out = procMetaGs(abr_path, mg_id, mg_sub_path, mg_raw_file_list, subcontig_path,
                             nthreads
                             )
    # Recruit subcontigs using OC-SVM
    covm_pass_dfs = []
    minhash_df['jacc_sim'] = minhash_df['jacc_sim'].astype(float)
    logging.info("Starting one-class SVM analysis\n")
    for sag_id in tqdm(set(minhash_df['sag_id'])):
        final_pass_df = recruitSubs(abr_path, sag_id, minhash_df, mg_covm_out)
        covm_pass_dfs.append(final_pass_df)
    covm_df = pd.concat(covm_pass_dfs)

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

    covm_max_list = []
    for sag_id in list(set(covm_recruit_filter_df['sag_id'])):
        sag_max_only_df = covm_recruit_filter_df.loc[covm_recruit_filter_df['sag_id'] == sag_id]
        covm_max_df = mg_tot_df[mg_tot_df['contig_id'].isin(list(sag_max_only_df['contig_id']))]
        covm_max_df['sag_id'] = sag_id
        covm_max_df = covm_max_df[['sag_id', 'subcontig_id', 'contig_id']]
        covm_max_list.append(covm_max_df)
    covm_final_max_df = pd.concat(covm_max_list)
    mh_covm_df = pd.concat([covm_final_max_df,
                            minhash_df[['sag_id', 'subcontig_id', 'contig_id']]
                            ])
    mh_covm_df.drop_duplicates(inplace=True)
    mh_covm_df.to_csv(o_join(abr_path, mg_id + '.abr_trimmed_recruits.tsv'),
                                    sep='\t', index=False
                                    )

    return mh_covm_df


def procMetaGs(abr_path, mg_id, mg_sub_path, mg_raw_file_list, subcontig_path, nthreads):
    # Build BWA index
    buildBWAindex(abr_path, mg_id, mg_sub_path)
    # Process each raw metagenome
    with open(mg_raw_file_list, 'r') as raw_fa_in:
        raw_data = raw_fa_in.readlines()
    sorted_bam_list = []
    for line in raw_data:
        raw_file_list = line.strip('\n').split('\t')
        # Run BWA mem
        pe_id = runBWAmem(abr_path, subcontig_path, mg_id, raw_file_list, nthreads)
        # Build/sorted .bam files
        mg_sort_out = runSamTools(abr_path, pe_id, nthreads, mg_id)
        sorted_bam_list.append(mg_sort_out)
    mg_covm_out = runCovM(abr_path, mg_id, nthreads, sorted_bam_list)

    return mg_covm_out



def buildBWAindex(abr_path, mg_id, mg_sub_path):
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

    return


def runBWAmem(abr_path, subcontig_path, mg_id, raw_file_list, nthreads):
    if len(raw_file_list) == 2:
        logging.info('[SABer]: Raw reads in FWD and REV file...\n')
        pe1 = raw_file_list[0]
        pe2 = raw_file_list[1]
        mem_cmd = ['bwa', 'mem', '-t', str(nthreads), '-p',
                   o_join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1, pe2
                   ] #TODO: add support for specifying number of threads
    else: # if the fastq is interleaved
        logging.info('[SABer]: Raw reads in interleaved file...\n')
        pe1 = raw_file_list[0]
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

    return pe_id


def runSamTools(abr_path, pe_id, nthreads, mg_id):
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

    return mg_sort_out


def runCovM(abr_path, mg_id, nthreads, sorted_bam_list):
    # run coverm on sorted bams
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

    return mg_covm_out


def recruitSubs(abr_path, sag_id, minhash_df, mg_covm_out):
    if isfile(o_join(abr_path, sag_id + '.abr_recruits.tsv')):
        #logging.info('[SABer]: Loading Abundance Recruits for  %s\n' % sag_id)
        final_pass_df = pd.read_csv(o_join(abr_path, sag_id + '.abr_recruits.tsv'),
                                    header=None,
                                    names=['sag_id', 'subcontig_id', 'contig_id'],
                                    sep='\t'
                                    )
    else:
        # subset df for sag_id
        minhash_sag_df = minhash_df.loc[minhash_df['sag_id'] == sag_id]
        minhash_90_list = list(minhash_sag_df['subcontig_id'].loc[
                                minhash_sag_df['jacc_sim_max'] >= 0.90]
                                )
        minhash_filter_df = minhash_sag_df.loc[(minhash_sag_df['jacc_sim_max'] == 1.0) &
                                               (minhash_sag_df['subcontig_recruits'] > 1)
                                               ]
        mh_jacc_list = list(set(minhash_filter_df['contig_id']))
        if len(mh_jacc_list) != 0:
            sag_mh_pass_df = minhash_df.loc[minhash_df['contig_id'].isin(mh_jacc_list)]
            overall_recruit_list = []
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

            keep_cols = [x for x in recruit_contigs_df.columns
                         if x.rsplit('.', 1)[1] != 'bam-var'
                         ]
            recruit_bam_df = recruit_contigs_df[keep_cols]
            nonrecruit_bam_df = nonrecruit_filter_df[keep_cols]
            final_pass_list = runOCSVM(recruit_contigs_df, nonrecruit_filter_df, sag_id)
            final_pass_df = pd.DataFrame(final_pass_list,
                                         columns=['sag_id', 'subcontig_id', 'contig_id']
                                         )
            final_pass_df.to_csv(o_join(abr_path, sag_id + '.abr_recruits.tsv'),
                                 header=False, index=False, sep='\t'
                                 )
            print("There are {} total subcontigs, {} contigs".format(
                  len(final_pass_df['subcontig_id']), len(final_pass_df['contig_id'].unique()))
                  )
            #logging.info("There are {} total subcontigs, {} contigs".format(
            #      len(final_pass_df['subcontig_id']), len(final_pass_df['contig_id'].unique()))
            #      )
        else:
            final_pass_df = pd.DataFrame([], columns=['sag_id', 'subcontig_id', 'contig_id'])

    return final_pass_df


def runOCSVM(sag_df, mg_df, sag_id):
    # fit OCSVM
    clf = svm.OneClassSVM(nu=0.9)
    clf.fit(sag_df.values)
    mg_pred = clf.predict(mg_df.values)
    mg_pred_df = pd.DataFrame(data=mg_pred, index=mg_df.index.values)
    svm_pass_df = mg_pred_df.loc[mg_pred_df[0] != -1]
    svm_pass_list = []
    #for md_nm in svm_pass_df.index.values:
    sub_mg_df = mg_df.loc[mg_df.index.isin(svm_pass_df.index.values)]
    for md_nm in sub_mg_df.index.values:
            svm_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])

    return svm_pass_list


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

    runAbundRecruiter(subcontig_path, abr_path, [mg_id, mg_sub_file],
                        minhash_recruit_df, per_pass, nthreads)

