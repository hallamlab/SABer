import argparse
import logging
from os.path import isfile, basename, getsize
from os.path import join as o_join
from subprocess import Popen

import pandas as pd
import saber.utilities as s_utils
from sklearn import svm

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
import multiprocessing
from sklearn.preprocessing import StandardScaler
import saber.tetranuc_recruiter as tra


def runAbundRecruiter(subcontig_path, abr_path, mg_sub_file, mg_raw_file_list,
                      minhash_df, covm_per_pass, nthreads, force
                      ):
    logging.info('Starting Abundance Recruitment\n')
    mg_id = mg_sub_file[0]
    if ((isfile(o_join(abr_path, mg_id + '.abr_trimmed_recruits.tsv'))) &
            (force is False)
    ):
        logging.info('Loading Abundance matrix for %s\n' % mg_id)
        mh_covm_df = pd.read_csv(o_join(abr_path, mg_id + '.abr_trimmed_recruits.tsv'), header=0,
                                 sep='\t'
                                 )
    else:
        mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
        mg_headers = tuple(mg_subcontigs.keys())
        logging.info('Building %s abundance table\n' % mg_id)
        mg_sub_path = o_join(subcontig_path, mg_id + '.subcontigs.fasta')
        # Process raw metagenomes to calculate abundances
        mg_covm_out = procMetaGs(abr_path, mg_id, mg_sub_path, mg_raw_file_list, subcontig_path,
                                 nthreads
                                 )
        # Recruit subcontigs using OC-SVM
        covm_pass_dfs = []
        minhash_df['jacc_sim'] = minhash_df['jacc_sim'].astype(float)
        logging.info("Starting one-class SVM analysis\n")
        covm_df = abund_recruiter(abr_path, covm_pass_dfs, mg_covm_out, minhash_df, nthreads)

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
        covm_recruit_filter_df = covm_recruit_df.loc[covm_recruit_df['subcontig_recruits'] != 0]
        # covm_recruit_filter_df = covm_recruit_df.loc[covm_recruit_df['percent_recruited'] >=
        #                                             float(covm_per_pass)
        #                                             ]

        covm_max_list = []
        for i, sag_id in enumerate(list(set(covm_recruit_filter_df['sag_id'])), 1):
            logging.info("\rSubsetting recruits for each SAG: {}/{}".format(i,
                                                                            len(list(set(
                                                                                covm_recruit_filter_df['sag_id'])))))
            sag_max_only_df = covm_recruit_filter_df.loc[covm_recruit_filter_df['sag_id'] == sag_id]
            covm_max_df = mg_tot_df[mg_tot_df['contig_id'].isin(list(sag_max_only_df['contig_id']))]
            covm_max_df['sag_id'] = sag_id
            covm_max_df = covm_max_df[['sag_id', 'subcontig_id', 'contig_id']]
            covm_max_list.append(covm_max_df)
        logging.info('\n')
        covm_final_max_df = pd.concat(covm_max_list)
        mh_covm_df = pd.concat([covm_final_max_df,
                                minhash_df[['sag_id', 'subcontig_id', 'contig_id']]
                                ])
        mh_covm_df.drop_duplicates(inplace=True)
        mh_covm_df.to_csv(o_join(abr_path, mg_id + '.abr_trimmed_recruits.tsv'),
                          sep='\t', index=False
                          )

    return mh_covm_df


def abund_recruiter(abr_path, covm_pass_dfs, mg_covm_out, minhash_df, nthreads):
    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    # Prep MinHash
    minhash_df.sort_values(by='jacc_sim', ascending=False, inplace=True)
    minhash_dedup_df = minhash_df[['sag_id', 'subcontig_id', 'contig_id', 'jacc_sim', 'jacc_sim_max']
    ]  # .drop_duplicates(subset=['sag_id', 'contig_id'])
    # ].loc[minhash_df['jacc_sim'] == 1.0].drop_duplicates(subset=['sag_id', 'contig_id'])
    mh_recruit_dict = tra.build_uniq_dict(minhash_dedup_df, 'sag_id', nthreads,
                                          'MinHash Recruits')  # TODO: this might not need multithreading
    for i, sag_id in enumerate(mh_recruit_dict.keys(), 1):
        if isfile(o_join(abr_path, sag_id + '.abr_recruits.tsv')):
            # logging.info('Loading Abundance Recruits for  %s\n' % sag_id)
            final_pass_df = pd.read_csv(o_join(abr_path, sag_id + '.abr_recruits.tsv'),
                                        header=None,
                                        names=['sag_id', 'subcontig_id', 'contig_id'],
                                        sep='\t'
                                        )
            covm_pass_dfs.append(final_pass_df)
        else:
            logging.info('\rPrepping for OCSVM: {}/{}'.format(i, len(mh_recruit_dict.keys())))
            # mh_sub_df = minhash_df.loc[minhash_df['sag_id'] == sag_id]  # TODO: this is slow, needs to be refactored
            mh_sub_df = mh_recruit_dict[sag_id]
            arg_list.append([abr_path, sag_id, mh_sub_df, mg_covm_out])
    logging.info('\n')
    logging.info("{} already complete, {} to run\n".format(len(covm_pass_dfs), len(arg_list)))
    results = pool.imap_unordered(recruitSubs, arg_list)
    for i, output in enumerate(results, 1):
        covm_pass_dfs.append(output)
        logging.info('\rRecruiting with Abundance Model: {}/{}'.format(i, len(arg_list)))
        # logging.info("\rThere are {} total subcontigs, {} contigs".format(
        #    len(output['subcontig_id']), len(output['contig_id'].unique()))
        # )
    logging.info('\n')
    pool.close()
    pool.join()
    covm_df = pd.concat(covm_pass_dfs)
    return covm_df


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
        pe_id, mg_sam_out = runBWAmem(abr_path, subcontig_path, mg_id, raw_file_list,
                                      nthreads
                                      )
        # Build/sorted .bam files
        mg_sort_out = runSamTools(abr_path, pe_id, nthreads, mg_id, mg_sam_out)
        sorted_bam_list.append(mg_sort_out)
    logging.info('\n')
    mg_covm_out = runCovM(abr_path, mg_id, nthreads, sorted_bam_list)

    return mg_covm_out


def buildBWAindex(abr_path, mg_id, mg_sub_path):
    index_ext_list = ['amb', 'ann', 'bwt', 'pac', 'sa']
    check_ind_list = ['.'.join([mg_sub_path, x]) for x in index_ext_list]
    if False in (isfile(f) for f in check_ind_list):
        # Use BWA to build an index for metagenome assembly
        logging.info('Creating index with BWA\n')
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
        logging.info('Raw reads in FWD and REV file...\n')
        pe1 = raw_file_list[0]
        pe2 = raw_file_list[1]
        mem_cmd = ['bwa', 'mem', '-t', str(nthreads), '-p',
                   o_join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1, pe2
                   ]  # TODO: add support for specifying number of threads
    else:  # if the fastq is interleaved
        logging.info('\rRaw reads in interleaved file...')
        pe1 = raw_file_list[0]
        mem_cmd = ['bwa', 'mem', '-t', str(nthreads), '-p',
                   o_join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1
                   ]  # TODO: how to get install path for executables?
    pe_basename = basename(pe1)
    pe_id = pe_basename.split('.')[0]
    # BWA sam file exists?
    mg_sam_out = o_join(abr_path, pe_id + '.sam')
    if isfile(mg_sam_out) == False:
        logging.info('Running BWA mem on %s\n' % pe_id)
        with open(mg_sam_out, 'w') as sam_file:
            with open(o_join(abr_path, pe_id + '.stderr.txt'), 'w') as stderr_file:
                run_mem = Popen(mem_cmd, stdout=sam_file, stderr=stderr_file)
                run_mem.communicate()

    return pe_id, mg_sam_out


def runSamTools(abr_path, pe_id, nthreads, mg_id, mg_sam_out):
    mg_bam_out = o_join(abr_path, pe_id + '.bam')
    if isfile(mg_bam_out) == False:
        logging.info('Converting SAM to BAM with SamTools\n')
        bam_cmd = ['samtools', 'view', '-S', '-b', '-@', str(nthreads), mg_sam_out]
        with open(mg_bam_out, 'w') as bam_file:
            with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
                run_bam = Popen(bam_cmd, stdout=bam_file, stderr=stderr_file)
                run_bam.communicate()
    # sort bam file
    mg_sort_out = o_join(abr_path, pe_id + '.sorted.bam')
    if isfile(mg_sort_out) == False:
        logging.info('Sort BAM with SamTools\n')
        sort_cmd = ['samtools', 'sort', '-@', str(nthreads), mg_bam_out, '-o', mg_sort_out]
        with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
            run_sort = Popen(sort_cmd, stderr=stderr_file)
            run_sort.communicate()

    return mg_sort_out


def runCovM(abr_path, mg_id, nthreads, sorted_bam_list):
    # run coverm on sorted bams
    mg_covm_out = o_join(abr_path, mg_id + '.metabat.tsv')
    try:  # if file exists but is empty
        covm_size = getsize(mg_covm_out)
    except:  # if file doesn't exist
        covm_size = -1
    if covm_size <= 0:
        logging.info('Calculate mean abundance and variance with CoverM\n')
        covm_cmd = ['coverm', 'contig', '-t', str(nthreads), '-m', 'metabat', '-b'
                    ]
        covm_cmd.extend(sorted_bam_list)
        with open(mg_covm_out, 'w') as covm_file:
            with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
                run_covm = Popen(covm_cmd, stdout=covm_file, stderr=stderr_file)
                run_covm.communicate()

    return mg_covm_out


def recruitSubs(p):
    abr_path, sag_id, minhash_sag_df, mg_covm_out = p
    minhash_filter_df = minhash_sag_df.copy()  #.loc[(minhash_sag_df['jacc_sim_max'] == 1.0)]  # TODO: maybe try 0.5 as well
    if len(minhash_filter_df['sag_id']) != 0:
        mg_covm_df = pd.read_csv(mg_covm_out, header=0, sep='\t', index_col=['contigName'])
        mg_covm_df.drop(columns=['contigLen', 'totalAvgDepth'], inplace=True)
        scale = StandardScaler().fit(mg_covm_df.values)
        scaled_data = scale.transform(mg_covm_df.values)
        std_merge_df = pd.DataFrame(scaled_data, index=mg_covm_df.index)
        recruit_contigs_df = std_merge_df.loc[std_merge_df.index.isin(
            list(minhash_filter_df['subcontig_id']))
        ]
        nonrecruit_filter_df = std_merge_df.copy()
        final_pass_list = runOCSVM(recruit_contigs_df, nonrecruit_filter_df, sag_id)
        final_pass_df = pd.DataFrame(final_pass_list,
                                     columns=['sag_id', 'subcontig_id', 'contig_id']
                                     )
        final_pass_df.to_csv(o_join(abr_path, sag_id + '.abr_recruits.tsv'),
                             header=False, index=False, sep='\t'
                             )
    else:
        final_pass_df = pd.DataFrame([], columns=['sag_id', 'subcontig_id', 'contig_id'])

    return final_pass_df


def runOCSVM(sag_df, mg_df, sag_id):
    # fit OCSVM
    clf = svm.OneClassSVM(nu=0.4, gamma=10)
    clf.fit(sag_df.values)
    mg_pred = clf.predict(mg_df.values)
    contig_id_list = [x.rsplit('_', 1)[0] for x in mg_df.index.values]
    pred_df = pd.DataFrame(zip(mg_df.index.values, contig_id_list, mg_pred),
                           columns=['subcontig_id', 'contig_id', 'ocsvm_pred']
                           )
    # key_cnts = pred_df.groupby('contig_id')['ocsvm_pred'].count().reset_index()
    key_cnts = pred_df.groupby('contig_id')['ocsvm_pred'].value_counts().reset_index(name='count')
    val_perc = pred_df.groupby('contig_id')['ocsvm_pred'].value_counts(
        normalize=True).reset_index(name='percent')
    pos_perc = val_perc.loc[val_perc['ocsvm_pred'] == 1]
    major_df = pos_perc.loc[pos_perc['percent'] >= 0.51]
    # pos_perc = key_cnts.loc[key_cnts['ocsvm_pred'] == 1]
    # major_df = pos_perc.loc[pos_perc['count'] != 0]
    major_pred = [1 if x in list(major_df['contig_id']) else -1
                  for x in contig_id_list
                  ]
    pred_df['major_pred'] = major_pred
    svm_pass_list = []
    # for md_nm in svm_pass_df.index.values:
    sub_mg_df = pred_df.loc[pred_df['major_pred'] == 1]
    for md_nm in sub_mg_df['subcontig_id']:
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
