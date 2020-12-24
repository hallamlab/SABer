import logging
import multiprocessing
from os.path import isfile
from os.path import join as o_join

import numpy as np
import pandas as pd
import saber.utilities as s_utils
import sourmash
from sourmash.sbtmh import SigLeaf

pd.set_option('display.max_columns', None)


def run_minhash_recruiter(sig_path, mhr_path, sag_sub_files, mg_sub_file,
                          jacc_threshold, mh_per_pass, nthreads, force
                          ):
    logging.info('Starting MinHash Recruitment\n')
    # Calculate/Load MinHash Signatures with SourMash for MG subseqs
    mg_id = mg_sub_file[0]
    mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
    mg_headers = tuple(mg_subcontigs.keys())

    if ((isfile(o_join(mhr_path, mg_id + '.mhr_trimmed_recruits.tsv'))) &
            (force is False)
    ):
        logging.info('MinHash already done, moving on (use --force to re-run)\n')
        minhash_filter_df = pd.read_csv(o_join(mhr_path, mg_id + '.mhr_trimmed_recruits.tsv'), header=0,
                                        sep='\t'
                                        )
    else:
        build_list, minhash_pass_list = sag_recruit_checker(mhr_path, sag_sub_files)
        if len(build_list) != 0:
            sag_sig_dict = build_sag_sig_dict(build_list, nthreads, sig_path)
            build_mg_sbt(mg_id, mg_sub_file, sig_path, nthreads, checkonly=True)  # make sure SBT exists first
            pool = multiprocessing.Pool(processes=nthreads)
            sbt_args = mg_id, mg_sub_file, sig_path, nthreads
            arg_list = []
            chunk_list = [list(x) for x in np.array_split(np.array(list(sag_sig_dict.keys())),
                                                          nthreads) if len(list(x)) != 0
                          ]  # TODO: might be a better way to chunk up the list?
            logging.info('Built {} Blocks of SAG Signature Sets\n'.format(len(chunk_list)))
            for i, sag_id_list in enumerate(chunk_list):
                sub_sag_sig_dict = {k: sag_sig_dict[k] for k in sag_sig_dict.keys()}
                arg_list.append([sbt_args, mhr_path, sag_id_list, sub_sag_sig_dict])
            results = pool.imap_unordered(compare_sag_sbt, arg_list)
            logging.info('Querying {} Signature Blocks against SBT\n'.format(len(chunk_list)))
            logging.info('WARNING: This can be VERY time consuming, be patient\n'.format(len(chunk_list)))
            for i, search_df in enumerate(results):
                logging.info('\rSignatures Queried Against SBT: {}/{}'.format(len(search_df),
                                                                              len(build_list))
                             )  # TODO: this doesn't print properly, needs to be fixed
                minhash_pass_list.extend(search_df)
            logging.info('\n')
            pool.close()
            pool.join()

        if len(minhash_pass_list) > 1:
            minhash_df = pd.concat(minhash_pass_list)
        else:
            minhash_df = minhash_pass_list[0]

        minhash_df['jacc_sim'] = minhash_df['jacc_sim'].astype(float)
        recruit_list = list(minhash_df['subcontig_id'].loc[minhash_df['jacc_sim'] >= 0.10])
        minhash_recruit_df = minhash_df.loc[minhash_df['subcontig_id'].isin(recruit_list)]
        logging.info('Compiling all MinHash Recruits\n')
        # Count # of subcontigs recruited to each SAG via sourmash
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
                                              on=['sag_id', 'subcontig_id', 'contig_id']
                                              )
        minhash_filter_df = merge_jacc_df.loc[((merge_jacc_df['jacc_sim_max'] > 0.5) &
                                               (merge_jacc_df['subcontig_recruits'] > 3)) |
                                              (merge_jacc_df['jacc_sim_max'] >= 0.9)
                                              ]
        minhash_filter_df.to_csv(o_join(mhr_path, mg_id + '.mhr_trimmed_recruits.tsv'), sep='\t',
                                 index=False
                                 )
        minhash_filter_df.to_csv(o_join(mhr_path, mg_id + '.mhr_trimmed_recruits.tsv'), sep='\t', index=False)
    logging.info('MinHash Recruitment Algorithm Complete\n')
    return minhash_filter_df


def build_sag_sig_dict(build_list, nthreads, sig_path):
    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for i, sag_rec in enumerate(build_list):
        sag_id, sag_file = sag_rec
        arg_list.append([sag_file, sag_id, sig_path])
    results = pool.imap_unordered(load_sag_sigs, arg_list)
    sag_sig_dict = {}
    for i, sag_sig_rec in enumerate(results):
        sag_id, sag_sig_list = sag_sig_rec
        logging.info('\rLoading/Building SAGs Signatures: {}/{}'.format(i + 1, len(build_list)))
        sag_sig_dict[sag_id] = sag_sig_list
    logging.info('\n')
    pool.close()
    pool.join()
    return sag_sig_dict


def compare_sag_sbt(p):  # TODO: needs stdout for user monitoring
    sbt_args, mhr_path, sag_id_list, sag_sig_dict = p
    mg_id, mg_sub_file, sig_path, nthreads = sbt_args
    mg_sbt = build_mg_sbt(mg_id, mg_sub_file, sig_path, nthreads)
    search_df_list = []
    for sag_id in sag_id_list:
        sag_sig_list = sag_sig_dict[sag_id]
        search_list = []
        for i, sig in enumerate(sag_sig_list):
            sbt_out = sourmash.search_sbt_index(mg_sbt, sig, threshold=0.1)
            for target in sbt_out:
                similarity = target[1]
                t_sig = target[0]
                q_subcontig = t_sig.name()
                q_contig = q_subcontig.rsplit('_', 1)[0]
                search_list.append([sag_id, q_subcontig, q_contig, similarity])
        search_df = pd.DataFrame(search_list, columns=['sag_id', 'subcontig_id', 'contig_id',
                                                       'jacc_sim'
                                                       ])
        search_df['jacc_sim'] = search_df['jacc_sim'].astype(float)
        search_df.sort_values(by='jacc_sim', ascending=False, inplace=True)
        search_df.drop_duplicates(subset='subcontig_id', inplace=True)
        search_df.to_csv(o_join(mhr_path, sag_id + '.mhr_recruits.tsv'), sep='\t',
                         index=False
                         )
        search_df_list.append(search_df)
    return search_df_list


def build_mg_sbt(mg_id, mg_sub_file, sig_path, nthreads, checkonly=False):
    mg_sbt_file = o_join(sig_path, mg_id + '.sbt.zip')
    if isfile(mg_sbt_file):
        if checkonly is True:
            logging.info('%s Sequence Bloom Tree Exists\n' % mg_id)
            mg_sbt_tree = True
        else:
            # logging.info('Loading %s Sequence Bloom Tree\n' % mg_id)
            mg_sbt_tree = sourmash.load_sbt_index(mg_sbt_file)
    else:
        logging.info('Building %s Sequence Bloom Tree\n' % mg_id)
        mg_sig_list = load_mg_sigs(mg_id, mg_sub_file, nthreads, sig_path)
        mg_sbt_tree = sourmash.create_sbt_index()
        pool = multiprocessing.Pool(processes=nthreads)
        results = pool.imap_unordered(build_leaf, mg_sig_list)
        leaf_list = []
        for i, leaf in enumerate(results):
            logging.info('\rBuilding leaves for SBT: {0:.0%}'.format((i + 1) / len(mg_sig_list)))
            leaf_list.append(leaf)
        logging.info('\n')
        for i, lef in enumerate(leaf_list):
            logging.info('\rAdding leaves to tree: {0:.0%}'.format((i + 1) / len(leaf_list)))
            mg_sbt_tree.add_node(lef)
        logging.info('\n')
        mg_sbt_tree.save(mg_sbt_file)
        pool.close()
        pool.join()
    return mg_sbt_tree


def build_leaf(sig):
    leaf = SigLeaf(sig.md5sum(), sig)
    return leaf


def load_mg_sigs(mg_id, mg_sub_file, nthreads, sig_path):
    if isfile(o_join(sig_path, mg_id + '.metaG.sig')):
        logging.info('Loading %s Signatures\n' % mg_id)
        mg_sig_list = tuple(sourmash.signature.load_signatures(o_join(sig_path, mg_id + \
                                                                      '.metaG.sig')
                                                               ))
    else:
        logging.info('Loading subcontigs for %s\n' % mg_id)
        mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
        mg_sig_list = build_mg_sigs(mg_id, mg_subcontigs, nthreads, sig_path)
    return mg_sig_list


def load_sag_sigs(p):
    sag_file, sag_id, sig_path = p
    if isfile(o_join(sig_path, sag_id + '.SAG.sig')):
        sag_sig_list = tuple(sourmash.signature.load_signatures(o_join(sig_path,
                                                                       sag_id + '.SAG.sig')
                                                                ))
    else:
        sag_sig_list = build_sag_sigs(sag_file, sag_id, sig_path)
    return sag_id, sag_sig_list


'''
def compare_sag_mg_sigs(jacc_threshold, mhr_path, nthreads, sag_id, sag_sig_list, sig_path, split_mg_sig_list):
    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for mg_sig_sub_list in split_mg_sig_list:
        arg_list.append([sag_id, sag_sig_list, mhr_path, sig_path, mg_sig_sub_list,
                         jacc_threshold
                         ])
    results = pool.imap_unordered(compare_sigs, arg_list)
    logging.info('Comparing Signature for %s\n' % sag_id)
    merge_list = []
    for i, o_list in enumerate(results):
        sys.stderr.write('\rdone {0:.0%}'.format(i / len(arg_list)))
        merge_list.extend(o_list)
    merge_df = pd.DataFrame(merge_list, columns=['sag_id', 'subcontig_id', 'contig_id',
                                                 'jacc_sim', 'recruit_bool'
                                                 ])
    merge_df['jacc_sim'] = merge_df['jacc_sim'].astype(float)
    merge_df.sort_values(by='jacc_sim', ascending=False, inplace=True)
    merge_df.drop_duplicates(subset='subcontig_id', inplace=True)
    merge_df.to_csv(o_join(mhr_path, sag_id + '.mhr_recruits.tsv'), sep='\t',
                    index=False
                    )
    pool.close()
    pool.join()
    return merge_df
'''


def build_sag_sigs(sag_file, sag_id, sig_path):
    sag_subcontigs = s_utils.get_seqs(sag_file)
    sag_headers = tuple(sag_subcontigs.keys())
    sag_sig_list = []
    for i, sag_head in enumerate(sag_headers):
        sag_sig = build_signature([sag_head, str(sag_subcontigs[sag_head].seq)])
        sag_sig_list.append(sag_sig)
    with open(o_join(sig_path, sag_id + '.SAG.sig'), 'w') as sag_out:
        sourmash.signature.save_signatures(sag_sig_list, fp=sag_out)
    sag_sig_list = tuple(sag_sig_list)
    return sag_sig_list


def build_mg_sigs(mg_id, mg_subcontigs, nthreads, sig_path):
    mg_headers = mg_subcontigs.keys()
    arg_list = []
    for i, mg_head in enumerate(mg_headers):
        arg_list.append([mg_head, str(mg_subcontigs[mg_head].seq)])
    logging.info('\n')
    pool = multiprocessing.Pool(processes=nthreads)
    results = pool.imap_unordered(build_signature, arg_list)
    mg_sig_list = []
    for i, mg_sig in enumerate(results):
        logging.info('\rBuilding MinHash Signatures for %s: {0:.0%} done'.format([mg_id, i / len(arg_list)]))
        mg_sig_list.append(mg_sig)
    logging.info('\n')
    pool.close()
    pool.join()
    with open(o_join(sig_path, mg_id + '.metaG.sig'), 'w') as mg_out:
        sourmash.signature.save_signatures(mg_sig_list, fp=mg_out)
    mg_sig_list = tuple(mg_sig_list)
    return mg_sig_list


def sag_recruit_checker(mhr_path, sag_sub_files):
    l = 0
    b = 0
    build_list = []
    minhash_pass_list = []
    for i, sag_rec in enumerate(sag_sub_files):
        sag_id, sag_file = sag_rec
        if isfile(o_join(mhr_path, sag_id + '.mhr_recruits.tsv')):
            # logging.info('Loading %s and MetaG signature recruit list\n' % sag_id)
            pass_df = pd.read_csv(o_join(mhr_path, sag_id + '.mhr_recruits.tsv'), header=0,
                                  sep='\t'
                                  )
            minhash_pass_list.append(pass_df)
            l += 1
        else:
            build_list.append(sag_rec)
            b += 1
        logging.info('\rChecking for previously completed SAGs: {}/{} done'.format(l, b))
    logging.info('\n')
    return build_list, minhash_pass_list


def build_signature(p):
    header, seq = p
    mg_minhash = sourmash.MinHash(n=10, ksize=51)  # , scaled=100)
    mg_minhash.add_sequence(str(seq), force=True)
    mg_sig = sourmash.SourmashSignature(mg_minhash, name=header)

    return mg_sig


'''
def compare_sigs(p):
    sag_id, sag_sig_list, mhr_path, sig_path, mg_sig_list, jacc_threshold = p
    # logging.info('Comparing  %s and MetaG signature\n' % sag_id)
    pass_list = []
    for sag_sig in sag_sig_list:
        for mg_sig in mg_sig_list:
            jacc_sim = mg_sig.similarity(sag_sig)
            mg_nm = mg_sig.name()
            if jacc_sim >= jacc_threshold:
                pass_list.append([sag_id, mg_nm, mg_nm.rsplit('_', 1)[0], str(jacc_sim), True])
    pass_list = tuple(pass_list)

    return pass_list
'''


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
