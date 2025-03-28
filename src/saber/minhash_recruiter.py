import logging
import multiprocessing
from os.path import isfile, getsize
from os.path import join as o_join

import numpy as np
import pandas as pd
from tqdm import tqdm
import sourmash
from sourmash.sbtmh import SigLeaf

import saber.utilities as s_utils

pd.set_option('display.max_columns', None)


def run_minhash_recruiter(sig_path, mhr_path, sag_sub_files, mg_sub_file, nthreads, min_len, kmer_size):
    logging.info('Starting MinHash Recruitment\n')
    # Calculate/Load MinHash Signatures with SourMash for MG subseqs
    mg_id = mg_sub_file[0]
    kmer_list = [kmer_size]
    mh_kmer_recruits_dict = {}
    for kmer in kmer_list:
        mh_recruit_file = o_join(mhr_path, mg_id + '.' + str(kmer) + '.mhr_contig_recruits.tsv')
        if isfile(mh_recruit_file) == False:
            build_list, minhash_pass_list = sag_recruit_checker(mhr_path, sag_sub_files, kmer)
            if len(build_list) != 0:
                sag_sig_dict = build_sag_sig_dict(build_list, nthreads, sig_path, kmer)
                build_mg_sbt(mg_id, mg_sub_file, sig_path, nthreads, kmer, min_len, checkonly=True)  # make sure SBT exists first
                pool = multiprocessing.Pool(processes=nthreads)
                sbt_args = mg_id, mg_sub_file, sig_path, nthreads
                chunk_list = [list(x) for x in np.array_split(np.array(list(sag_sig_dict.keys())),
                                                              nthreads * 10) if len(list(x)) != 0
                              ]  # TODO: might be a better way to chunk up the list?
                logging.info('Built {} Blocks of Trusted Contigs Signature Sets\n'.format(len(chunk_list)))
                arg_list = []
                for i, sag_id_list in enumerate(chunk_list):
                    sub_sag_sig_dict = {k: sag_sig_dict[k] for k in sag_id_list}
                    arg_list.append([sbt_args, mhr_path, sag_id_list, sub_sag_sig_dict, kmer, min_len])
                results = pool.imap_unordered(compare_sag_sbt, arg_list)
                logging.info('Querying {} Signature Blocks against SBT\n'.format(len(chunk_list)))
                logging.info('WARNING: This can be VERY time consuming, '
                             'be patient\n'.format(len(chunk_list))
                             )
                df_cnt = 0
                logging.info('Signatures Queried Against SBT:\n')
                for search_df in tqdm(results):
                    df_cnt += len(search_df)
                    minhash_pass_list.extend(search_df)
                pool.close()
                pool.join()

            if len(minhash_pass_list) > 1:
                minhash_df = pd.concat(minhash_pass_list)
            else:
                minhash_df = minhash_pass_list[0]

            minhash_df['jacc_sim'] = minhash_df['jacc_sim'].astype(float)
            minhash_recruit_df = minhash_df.copy()
            minhash_recruit_df.to_csv(mh_recruit_file, sep='\t', index=False)
        else:
            logging.info('MinHash already done\n')
        mh_recruit_df = pd.read_csv(mh_recruit_file, header=0, sep='\t')
        mh_kmer_recruits_dict[kmer] = mh_recruit_df
    logging.info('Cleaning up intermediate files...\n')
    for s in ["*.sig", "*.mhr_recruits.tsv", "*.sbt.zip"]:
        s_utils.runCleaner(mhr_path, s)

    logging.info('MinHash Recruitment Algorithm Complete\n')
    return mh_kmer_recruits_dict


def build_sag_sig_dict(build_list, nthreads, sig_path, kmer):
    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for i, sag_rec in enumerate(build_list):
        sag_id, sag_file = sag_rec
        arg_list.append([sag_file, sag_id, sig_path, kmer])
    results = pool.imap_unordered(load_sag_sigs, arg_list)
    sag_sig_dict = {}
    logging.info('Loading/Building Trusted Contig Signatures:\n')
    for sag_sig_rec in tqdm(results):
        sag_id, sag_sig_list = sag_sig_rec
        sag_sig_dict[sag_id] = sag_sig_list
    pool.close()
    pool.join()

    return sag_sig_dict


def compare_sag_sbt(p):  # TODO: needs stdout for user monitoring
    sbt_args, mhr_path, sag_id_list, sag_sig_dict, kmer, min_len = p
    mg_id, mg_sub_file, sig_path, nthreads = sbt_args
    mg_sbt = build_mg_sbt(mg_id, mg_sub_file, sig_path, nthreads, kmer, min_len)
    search_df_list = []
    for i, sag_id in enumerate(sag_id_list):
        sag_sig_list = sag_sig_dict[sag_id]
        search_list = []
        for i, sig in enumerate(sag_sig_list):
            sbt_out = mg_sbt.search(sig, threshold=0.000000000001)
            sbt_out_cont = mg_sbt.search(sig, threshold=0.000000000001, do_containment=True)
            sbt_out.extend(sbt_out_cont)
            r_contig = sig
            for similarity, t_sig, filename in sbt_out:
                q_contig = t_sig
                search_list.append([sag_id, r_contig, q_contig, similarity])
        search_df = pd.DataFrame(search_list, columns=['sag_id', 'r_contig_id', 'q_contig_id',
                                                       'jacc_sim'
                                                       ])
        search_df['jacc_sim'] = search_df['jacc_sim'].astype(float)
        search_df.sort_values(by='jacc_sim', ascending=False, inplace=True)
        search_file = o_join(mhr_path, sag_id + '.' + str(kmer) + '.mhr_recruits.tsv')
        search_df.to_csv(search_file, sep='\t',
                         index=False
                         )
        search_df_list.append(search_df)
    return search_df_list


def build_mg_sbt(mg_id, mg_sub_file, sig_path, nthreads, kmer, min_len, checkonly=False):
    mg_sbt_file = o_join(sig_path, mg_id + '.' + str(kmer) + '.sbt.zip')
    if isfile(mg_sbt_file):
        if checkonly is True:
            logging.info('%s Sequence Bloom Tree Exists\n' % mg_id)
            mg_sbt_tree = True
        else:
            mg_sbt_tree = sourmash.load_sbt_index(mg_sbt_file)
    else:
        logging.info('Building %s Sequence Bloom Tree\n' % mg_id)  # TODO: perhaps multiple smaller SBTs would be better
        mg_sig_list = load_mg_sigs(mg_id, mg_sub_file, nthreads, sig_path, kmer, min_len)
        mg_sbt_tree = sourmash.create_sbt_index()
        pool = multiprocessing.Pool(processes=nthreads)
        results = pool.imap_unordered(build_leaf, mg_sig_list)
        leaf_list = []
        logging.info('Building leaves for SBT:\n')
        for leaf in tqdm(results):
            leaf_list.append(leaf)
        leaf_list = tuple(leaf_list)
        logging.info('Adding leaves to tree:\n')
        for lef in tqdm(leaf_list):
            mg_sbt_tree.add_node(lef)
        mg_sbt_tree.save(mg_sbt_file)
        pool.close()
        pool.join()
        mg_sbt_tree = None

    return mg_sbt_tree


def build_leaf(sig):
    leaf = SigLeaf(sig.md5sum(), sig)
    return leaf


def load_mg_sigs(mg_id, mg_sub_file, nthreads, sig_path, kmer, min_len):
    if isfile(o_join(sig_path, mg_id + '.' + str(kmer) + '.metaG.sig')):
        logging.info('Loading %s Signatures\n' % mg_id)
        mg_sig_list = tuple(sourmash.signature.load_signatures(o_join(sig_path, mg_id + \
                                                                      '.' + str(kmer) + '.metaG.sig')
                                                               ))
    else:
        logging.info('Loading subcontigs for %s\n' % mg_id)
        mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
        mg_sig_list = build_mg_sigs(mg_id, mg_subcontigs, nthreads, sig_path, kmer, min_len)
    return mg_sig_list


def load_sag_sigs(p):
    sag_file, sag_id, sig_path, kmer = p
    if isfile(o_join(sig_path, sag_id + '.' + str(kmer) + '.TC.sig')):
        sag_sig_list = tuple(sourmash.signature.load_signatures(o_join(sig_path,
                                                                       sag_id + '.' + str(kmer) + '.TC.sig')
                                                                ))
    else:
        sag_sig_list = build_sag_sigs(sag_file, sag_id, sig_path, kmer)
    return sag_id, sag_sig_list


def build_sag_sigs(sag_file, sag_id, sig_path, kmer):
    sag_subcontigs = s_utils.get_seqs(sag_file)
    sag_sig_list = []
    for i, sag_rec in enumerate(sag_subcontigs):
        sag_sig = build_signature([sag_rec[0], str(sag_rec[1]), kmer])
        sag_sig_list.append(sag_sig)
    with open(o_join(sig_path, sag_id + '.' + str(kmer) + '.TC.sig'), 'w') as sag_out:
        sourmash.signature.save_signatures(sag_sig_list, fp=sag_out)
    sag_sig_list = tuple(sag_sig_list)
    return sag_sig_list


def build_mg_sigs(mg_id, mg_subcontigs, nthreads, sig_path, kmer, min_len):
    arg_list = []
    for i, mg_rec in enumerate(mg_subcontigs):
        if len(str(mg_rec[1])) >= min_len:
            arg_list.append([mg_rec[0], str(mg_rec[1]), kmer])
    pool = multiprocessing.Pool(processes=nthreads)
    results = pool.imap_unordered(build_signature, arg_list)
    mg_sig_list = []
    logging.info('Building MinHash Signatures for metagenome contigs:\n')
    for mg_sig in tqdm(results):
        mg_sig_list.append(mg_sig)
    pool.close()
    pool.join()
    with open(o_join(sig_path, mg_id + '.' + str(kmer) + '.metaG.sig'), 'w') as mg_out:
        sourmash.signature.save_signatures(mg_sig_list, fp=mg_out)
    mg_sig_list = tuple(sourmash.signature.load_signatures(o_join(sig_path, mg_id + '.' + str(kmer) + '.metaG.sig')))

    return mg_sig_list


def sag_recruit_checker(mhr_path, sag_sub_files, kmer):
    build_list = []
    minhash_pass_list = []
    l = 0
    b = 0
    logging.info('Checking for previously completed Trusted Contigs:\n')
    for sag_rec in tqdm(sag_sub_files):
        sag_id, sag_file = sag_rec
        mh_file = o_join(mhr_path, sag_id + '.' + str(kmer) + '.mhr_recruits.tsv')
        if isfile(mh_file):
            filesize = getsize(mh_file)
        else:
            filesize = 0
        if filesize != 0:
            pass_df = pd.read_csv(mh_file, header=0, sep='\t')
            minhash_pass_list.append(pass_df)
            l += 1
        else:
            build_list.append(sag_rec)
            b += 1
    return build_list, minhash_pass_list


def build_signature(p):
    header, seq, kmer = p
    mg_minhash = sourmash.MinHash(ksize=kmer, scaled=1000, n=0)
    mg_minhash.add_sequence(str(seq), force=True)
    mg_sig = sourmash.SourmashSignature(mg_minhash, name=header)

    return mg_sig
