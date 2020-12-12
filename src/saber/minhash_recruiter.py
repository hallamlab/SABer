import logging
import sourmash
from os.path import isfile
from os.path import join as o_join
import pandas as pd
import saber.utilities as s_utils
import multiprocessing
import sys
from psutil import virtual_memory
import numpy as np
from tqdm import tqdm
from subprocess import Popen
import time

pd.set_option('display.max_columns', None)

def build_signature(p):
    header, seq = p
    mg_minhash = sourmash.MinHash(n=0, ksize=51, scaled=100)
    mg_minhash.add_sequence(str(seq), force=True)
    mg_sig = sourmash.SourmashSignature(mg_minhash, name=header)

    return mg_sig


def compare_sigs(p):
    sag_id, sag_sig_list, mhr_path, sig_path, mg_sig_list, jacc_threshold = p
    #sag_subcontigs = s_utils.get_seqs(sag_file)
    #sag_headers = tuple(sag_subcontigs.keys())

    # Calculate\Load MinHash Signatures with SourMash for SAG subseqs
    '''
    if isfile(o_join(sig_path, sag_id + '.SAG.sig')):
        logging.info('[SABer]: Loading Signature for %s\n' % sag_id)
        sag_sig = sourmash.signature.load_one_signature(o_join(sig_path,
                                                             sag_id + '.SAG.sig')
                                                        )
    else:
        logging.info('[SABer]: Building Signature for %s\n' % sag_id)
        sm_cmd = ['sourmash', 'compute', '-k', '51', '-n', '0', '--scaled', '100',
                    sag_file, '-o', o_join(sig_path, sag_id + '.SAG.sig')
                        ]
        with open(o_join(sig_path, sag_id + '.out.txt'), 'w') as sm_out:
                with open(o_join(sig_path, sag_id + '.err.txt'), 'w') as stderr_file:
                    run_sm = Popen(sm_cmd, stdout=sm_out, stderr=stderr_file)
                    run_sm.communicate()
        sag_sig = sourmash.signature.load_one_signature(o_join(sig_path,
                                                             sag_id + '.SAG.sig')
                                                        )
    '''
    '''
        sag_minhash = sourmash.MinHash(n=0, ksize=51, scaled=100)
        for sg_head in sag_headers:
            sag_subseq = str(sag_subcontigs[sg_head].seq)
            sag_minhash.add_sequence(sag_subseq, force=True)
        sag_sig = sourmash.SourmashSignature(sag_minhash, name=sag_id)
        with open(o_join(sig_path, sag_id + '.SAG.sig'), 'w') as sags_out:
            sourmash.signature.save_signatures([sag_sig], fp=sags_out)
    '''
    #logging.info('[SABer]: Comparing  %s and MetaG signature\n' % sag_id)
    pass_list = []
    for sag_sig in sag_sig_list:
        for mg_sig in mg_sig_list:
            jacc_sim = mg_sig.similarity(sag_sig)
            mg_nm = mg_sig.name()
            if jacc_sim >= jacc_threshold:
                pass_list.append([sag_id, mg_nm, mg_nm.rsplit('_', 1)[0], str(jacc_sim), True])
            #else:
            #    pass_list.append([sag_id, mg_nm, mg_nm.rsplit('_', 1)[0], str(jacc_sim), 'False'])
    pass_list = tuple(pass_list)

    return pass_list


def run_minhash_recruiter(sig_path, mhr_path, sag_sub_files, mg_sub_file,
                            jacc_threshold, mh_per_pass, nthreads, forcerun
                            ):
    logging.info('[SABer]: MinHash Recruitment Algorithm\n')
    # Calculate/Load MinHash Signatures with SourMash for MG subseqs
    mg_id = mg_sub_file[0]
    if ((isfile(o_join(mhr_path, mg_id + '.mhr_trimmed_recruits.tsv'))) &
        (forcerun == 'False')
        ):
        minhash_filter_df = pd.read_csv(o_join(mhr_path, mg_id + '.mhr_trimmed_recruits.tsv'), header=0,
                                        sep='\t'
                                        )
    else:
        logging.info('[SABer]: Loading subcontigs for %s\n' % mg_id)
        mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
        mg_headers = tuple(mg_subcontigs.keys())
        # Load comparisons OR Compare SAG sigs to MG sigs to find containment
        logging.info('[SABer]: Comparing Signatures of SAGs to MetaG contigs\n')
        l = 0
        b = 0
        build_list = []
        minhash_pass_list = []
        for i, sag_rec in enumerate(sag_sub_files):
            sag_id, sag_file = sag_rec
            if isfile(o_join(mhr_path, sag_id + '.mhr_recruits.tsv')):
                #logging.info('[SABer]: Loading %s and MetaG signature recruit list\n' % sag_id)
                pass_df = pd.read_csv(o_join(mhr_path, sag_id + '.mhr_recruits.tsv'), header=0,
                                      sep='\t'
                                      )
                minhash_pass_list.append(pass_df)
                l+=1
            else:
                build_list.append(sag_rec)
                b+=1
            logging.info('\r[SABer]: Loading/Comparing SAG and MetaG signatures: {}/{} done'.format(l, b))
        logging.info('\n')
        if b != 0:
            if isfile(o_join(sig_path, mg_id + '.metaG.sig')):  # TODO: MG should only be loaded if required
                logging.info('[SABer]: Loading %s Signatures\n' % mg_id)
                mg_sig_list = tuple(sourmash.signature.load_signatures(o_join(sig_path, mg_id + \
                                                                              '.metaG.sig')
                                                                                ))
            else:
                logging.info('[SABer]: Building Signatures for %s\n' % mg_id)
                pool = multiprocessing.Pool(processes=nthreads) # TODO: convert to Ray
                arg_list = []
                for i, mg_head in enumerate(mg_headers):
                    logging.info('\r[SABer]: Building multi-pool: {0:.0%} done'.format(i/len(mg_subcontigs)))
                    arg_list.append([mg_head, str(mg_subcontigs[mg_head].seq)])
                logging.info('\n')
                results = pool.imap_unordered(build_signature, arg_list)
                logging.info('\r[SABer]: Executing multi-pool:')
                mg_sig_list = []
                for i, mg_sig in enumerate(results):
                    logging.info('\r[SABer]: Executing multi-pool: {0:.0%} done'.format(i/len(arg_list)))
                    mg_sig_list.append(mg_sig)
                logging.info('\n')
                pool.close()
                pool.join()

                with open(o_join(sig_path, mg_id + '.metaG.sig'), 'w') as mg_out:
                    sourmash.signature.save_signatures(mg_sig_list, fp=mg_out)
                mg_sig_list = tuple(mg_sig_list)

            logging.info('[SABer]: Building/Comparing {} SAG signatures\n'.format(b))
            split_mg_sig_list = [x for x in np.array_split(mg_sig_list, nthreads, axis=0)]
            logging.info('[SABer]: Analyzing MinHash signatures\n')
            for sag_rec in tqdm(build_list):
                sag_id, sag_file = sag_rec
                if isfile(o_join(sig_path, sag_id + '.SAG.sig')):
                    logging.info('[SABer]: Loading Signature for %s\n' % sag_id)
                    sag_sig_list = tuple(sourmash.signature.load_signatures(o_join(sig_path,
                                                                         sag_id + '.SAG.sig')
                                                                    ))
                else:
                    logging.info('[SABer]: Building Signatures for %s\n' % sag_id)
                    sag_subcontigs = s_utils.get_seqs(sag_file)
                    sag_headers = tuple(sag_subcontigs.keys())
                    pool = multiprocessing.Pool(processes=nthreads) # TODO: convert to Ray
                    arg_list = []
                    for i, sag_head in enumerate(sag_headers):
                        logging.info('\r[SABer]: Building multi-pool: {0:.0%} done'.format(i/len(sag_subcontigs)))
                        arg_list.append([sag_head, str(sag_subcontigs[sag_head].seq)])
                    logging.info('\n')
                    results = pool.imap_unordered(build_signature, arg_list)
                    logging.info('\r[SABer]: Executing multi-pool:')
                    sag_sig_list = []
                    for i, sag_sig in enumerate(results):
                        logging.info('\r[SABer]: Executing multi-pool: {0:.0%} done'.format(i/len(arg_list)))
                        sag_sig_list.append(sag_sig)
                    logging.info('\n')
                    pool.close()
                    pool.join()

                    with open(o_join(sig_path, sag_id + '.SAG.sig'), 'w') as sag_out:
                        sourmash.signature.save_signatures(sag_sig_list, fp=sag_out)
                    sag_sig_list = tuple(sag_sig_list)

                    '''
                    sm_cmd = ['sourmash', 'compute', '--singleton', '-k', '51', '-n', '0',
                                '--scaled', '100', sag_file, '-o',
                                o_join(sig_path, sag_id + '.SAG.sig')
                                    ]
                    with open(o_join(sig_path, sag_id + '.out.txt'), 'w') as sm_out:
                            with open(o_join(sig_path, sag_id + '.err.txt'), 'w') as stderr_file:
                                run_sm = Popen(sm_cmd, stdout=sm_out, stderr=stderr_file)
                                run_sm.communicate()
                    sag_sig_list = tuple(sourmash.signature.load_signatures(o_join(sig_path,
                                                                         sag_id + '.SAG.sig')
                                                                    ))
                    '''

                pool = multiprocessing.Pool(processes=nthreads)
                arg_list = []
                for mg_sig_sub_list in split_mg_sig_list:
                    arg_list.append([sag_id, sag_sig_list, mhr_path, sig_path, mg_sig_sub_list,
                                     jacc_threshold
                                     ])
                results = pool.imap_unordered(compare_sigs, arg_list)
                logging.info('[SABer]: Comparing Signature for %s\n' % sag_id)
                merge_list = []
                for i, o_list in enumerate(results):
                    sys.stderr.write('\rdone {0:.0%}'.format(i/len(arg_list)))
                    merge_list.extend(o_list)
                merge_df = pd.DataFrame(merge_list, columns=['sag_id', 'subcontig_id', 'contig_id',
                                                              'jacc_sim', 'recruit_bool'
                                                              ])
                print(merge_df.head())
                merge_df.sort_values(by='jacc_sim', ascending=False, inplace=True)
                merge_df.drop_duplicates(subset='subcontig_id', inplace=True)
                merge_df.to_csv(o_join(mhr_path, sag_id + '.mhr_recruits.tsv'), sep='\t',
                                 index=False
                                 )
                pool.close()
                pool.join()
                minhash_pass_list.append(merge_df)

        if len(minhash_pass_list) > 1:
            minhash_df = pd.concat(minhash_pass_list)
        else:
            minhash_df = minhash_pass_list[0]
        minhash_df['jacc_sim'] = minhash_df['jacc_sim'].astype(float)
        recruit_list = list(minhash_df['subcontig_id'].loc[minhash_df['jacc_sim'] >= 0.10])
        minhash_recruit_df = minhash_df.loc[minhash_df['subcontig_id'].isin(recruit_list)]
        logging.info('[SABer]: Compiling all MinHash Recruits\n')
        # Count # of subcontigs recruited to each SAG via samsum
        mh_cnt_df = minhash_recruit_df.groupby(['sag_id', 'contig_id'])['subcontig_id'].count().reset_index()
        mh_cnt_df.columns = ['sag_id', 'contig_id', 'subcontig_recruits']
        mh_avg_df = minhash_recruit_df.groupby(['sag_id', 'contig_id'])['jacc_sim'].mean().reset_index()
        mh_avg_df.columns = ['sag_id', 'contig_id', 'jacc_sim_avg']
        mh_max_df = minhash_recruit_df.groupby(['sag_id', 'contig_id'])['jacc_sim'].max().reset_index()
        mh_max_df.columns = ['sag_id', 'contig_id', 'jacc_sim_max']


        # Build subcontig count for each MG contig
        mg_contig_list = [x.rsplit('_', 1)[0] for x in mg_headers]
        mg_tot_df = pd.DataFrame(zip(mg_contig_list, mg_headers),
                                 columns=['contig_id', 'subcontig_id'])
        mg_tot_cnt_df = mg_tot_df.groupby(['contig_id']).count().reset_index()
        mg_tot_cnt_df.columns = ['contig_id', 'subcontig_total']
        mh_recruit_df = mh_cnt_df.merge(mg_tot_cnt_df, how='left', on='contig_id')
        mh_recruit_df['percent_recruited'] = mh_recruit_df['subcontig_recruits'] / \
                                               mh_recruit_df['subcontig_total']
        mh_jacc_merge_df = mh_recruit_df.merge(mh_avg_df, how='left', on=['sag_id', 'contig_id'])
        mh_max_merge_df = mh_jacc_merge_df.merge(mh_max_df, how='left', on=['sag_id', 'contig_id'])
        #mh_recruit_filter_df = mh_max_merge_df.loc[(mh_max_merge_df['jacc_sim_avg'] == 1.0) |
        #                                            ((mh_max_merge_df['jacc_sim_avg'] >= 0.10) &
        #                                             (mh_max_merge_df['percent_recruited'] >= 0.10)
        #                                              )]
        # Only pass contigs that have the magjority of subcontigs recruited, i.e., >= 51%
        '''
        mh_recruit_filter_df = mh_recruit_df.loc[mh_recruit_df['percent_recruited'] >=
                                                     float(mh_per_pass)
                                                     ]
        mg_contig_per_max_df = mh_recruit_filter_df.groupby(['contig_id'])[
            'percent_recruited'].max().reset_index()
        mg_contig_per_max_df.columns = ['contig_id', 'percent_max']
        mh_recruit_max_df = mh_recruit_filter_df.merge(mg_contig_per_max_df, how='left',
                                                           on='subcontig_id')

        # Now pass contigs that have the maximum recruit % of subcontigs
        mh_max_only_df = mh_recruit_max_df.loc[mh_recruit_max_df['percent_recruited'] >=
                                                   mh_recruit_max_df['percent_max']
                                                   ]
        '''
        mh_max_list = []
        for sag_id in list(set(minhash_df['sag_id'])):
            sag_max_only_df = mh_max_merge_df.loc[mh_max_merge_df['sag_id'] == sag_id]
            mh_max_df = mg_tot_df[mg_tot_df['contig_id'].isin(list(sag_max_only_df['contig_id']))]
            mh_max_df['sag_id'] = sag_id
            sag_merge_df = mh_max_df.merge(sag_max_only_df, how='left',
                                           on=['contig_id', 'sag_id']
                                           )
            sag_merge_df = sag_merge_df[['sag_id', 'subcontig_id', 'contig_id', 'subcontig_recruits',
                                         'subcontig_total', 'percent_recruited', 'jacc_sim_avg',
                                         'jacc_sim_max'
                                         ]]
            mh_max_list.append(sag_merge_df)
        mh_final_max_df = pd.concat(mh_max_list)
        merge_jacc_df = mh_final_max_df.merge(minhash_df, how='left',
                                              on=['sag_id', 'subcontig_id','contig_id']
                                              )

        minhash_filter_df = merge_jacc_df.loc[((merge_jacc_df['jacc_sim_max'] >= 0.40) &
                                               (merge_jacc_df['subcontig_recruits'] > 3)) |
                                              (merge_jacc_df['jacc_sim_max'] >= 0.99)
                                              ] # |
                                              #(merge_jacc_df['percent_recruited'] >= 0.25) |
                                              #(merge_jacc_df['jacc_sim_avg'] >= 0.25)
                                              #]

        minhash_filter_df.to_csv(o_join(mhr_path, mg_id + '.mhr_trimmed_recruits.tsv'), sep='\t',
                            index=False
                            )
    logging.info('[SABer]: MinHash Recruitment Algorithm Complete\n')

    return minhash_filter_df
