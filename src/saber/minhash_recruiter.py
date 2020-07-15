import logging
import sourmash
from os.path import isfile
from os.path import join as o_join
import pandas as pd
import saber.utilities as s_utils
import multiprocessing
import ray
import sys
from psutil import virtual_memory
import numpy as np
from tqdm import tqdm
from subprocess import Popen



def build_signature(p):
    header, seq = p
    mg_minhash = sourmash.MinHash(n=0, ksize=51, scaled=100)
    mg_minhash.add_sequence(str(seq), force=True)
    mg_sig = sourmash.SourmashSignature(mg_minhash, name=header)

    return mg_sig


@ray.remote
def compare_sigs(sag_id, sag_file, mhr_path, sig_path, mg_sig_list, jacc_threshold):
    #sag_subcontigs = s_utils.get_seqs(sag_file)
    #sag_headers = tuple(sag_subcontigs.keys())

    # Calculate\Load MinHash Signatures with SourMash for SAG subseqs
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
        sag_minhash = sourmash.MinHash(n=0, ksize=51, scaled=100)
        for sg_head in sag_headers:
            sag_subseq = str(sag_subcontigs[sg_head].seq)
            sag_minhash.add_sequence(sag_subseq, force=True)
        sag_sig = sourmash.SourmashSignature(sag_minhash, name=sag_id)
        with open(o_join(sig_path, sag_id + '.SAG.sig'), 'w') as sags_out:
            sourmash.signature.save_signatures([sag_sig], fp=sags_out)
        '''
    logging.info('[SABer]: Comparing  %s and MetaG signature\n' % sag_id)
    pass_list = []
    for mg_sig in mg_sig_list:
        jacc_sim = mg_sig.similarity(sag_sig)
        mg_nm = mg_sig.name()
        if jacc_sim >= jacc_threshold:
            pass_list.append([sag_id, mg_nm, mg_nm.rsplit('_', 1)[0]])

    pass_list = tuple(pass_list)

    return pass_list


def run_minhash_recruiter(sig_path, mhr_path, sag_sub_files, mg_sub_file,
                            jacc_threshold, mh_per_pass, nthreads
                            ):
    logging.info('[SABer]: MinHash Recruitment Algorithm\n')
    # Calculate/Load MinHash Signatures with SourMash for MG subseqs
    mg_id = mg_sub_file[0]
    logging.info('[SABer]: Loading subcontigs for %s\n' % mg_id)
    mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
    mg_headers = tuple(mg_subcontigs.keys())
    #mg_id, mg_headers, mg_subs = mg_subcontigs
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
            with open(o_join(mhr_path, sag_id + '.mhr_recruits.tsv'), 'r') as mhr_in:
                pass_list = tuple([x.rstrip('\n').split('\t') for x in mhr_in.readlines()])
                minhash_pass_list.extend(pass_list)
            l+=1
        else:
            build_list.append(sag_rec)
            b+=1
        logging.info('\r[SABer]: Loading/Comparing SAG and MetaG signatures: {}/{} done'.format(l, b))
    logging.info('\n')
    if b != 0:
        logging.info('[SABer]: Building/Comparing {} SAG signatures\n'.format(b))
        logging.info('[SABer]: Initializing Ray cluster and Loading shared data\n')
        max_mem = int(virtual_memory().total*0.25)
        ray.init(num_cpus=nthreads, memory=max_mem, object_store_memory=max_mem)
        split_mg_sig_list = [ray.put(x) for x in np.array_split(mg_sig_list, nthreads, axis=0)]
        r_mhr_path = ray.put(mhr_path)
        r_jacc_threshold = ray.put(jacc_threshold)
        logging.info('[SABer]: Analyzing MinHash signatures\n')
        for sag_rec in tqdm(build_list):
            sag_id, sag_file = sag_rec
            #logging.info('[SABer]: Analyzing {} signature\n'.format(sag_id))
            futures = []
            for mg_sig_sub_list in split_mg_sig_list:
                futures.append(compare_sigs.remote(sag_id, sag_file, r_mhr_path, sig_path,
                                                    mg_sig_sub_list, r_jacc_threshold
                                                    ))
                #logging.info('\r[SABer]: Building execute list: {0:.0%} complete'.format((j+1)/len(split_mg_sig_list)))
            #logging.info('\n')
            ray_results = ray.get(futures)
            ray_results = [r for l in ray_results for r in l] # flatten list
            with open(o_join(mhr_path, sag_id + '.mhr_recruits.tsv'), 'w') as mhr_out:
                mhr_out.write('\n'.join(['\t'.join(x) for x in ray_results]))
        minhash_pass_list.extend(ray_results)
        ray.shutdown()
        #for g in get_futures:
        #    ray_results[g[0]].extend(g[1])
        #    logging.info('\r[SABer]: Comparison {0:.0%} complete'.format(i/len(build_list)))
        #logging.info('\n')
        '''
        for k in ray_results.keys():
            print(k)
            sag_rec_list = futures[k]
            with open(o_join(mhr_path, sag_id + '.mhr_recruits.tsv'), 'w') as mhr_out:
                mhr_out.write('\n'.join(['\t'.join(x) for x in sag_rec_list]))

        minhash_pass_list.extend(ray_results)
        '''
    minhash_df = pd.DataFrame(minhash_pass_list, columns=['sag_id', 'subcontig_id',
                                                          'contig_id'
                                                          ])
    logging.info('[SABer]: Compiling all MinHash Recruits\n')
    # Count # of subcontigs recruited to each SAG via samsum
    mh_cnt_df = minhash_df.groupby(['sag_id', 'contig_id']).count().reset_index()
    mh_cnt_df.columns = ['sag_id', 'contig_id', 'subcontig_recruits']
    # Build subcontig count for each MG contig
    mg_contig_list = [x.rsplit('_', 1)[0] for x in mg_headers]
    mg_tot_df = pd.DataFrame(zip(mg_contig_list, mg_headers),
                             columns=['contig_id', 'subcontig_id'])
    mg_tot_cnt_df = mg_tot_df.groupby(['contig_id']).count().reset_index()
    mg_tot_cnt_df.columns = ['contig_id', 'subcontig_total']
    mh_recruit_df = mh_cnt_df.merge(mg_tot_cnt_df, how='left', on='contig_id')
    mh_recruit_df['percent_recruited'] = mh_recruit_df['subcontig_recruits'] / \
                                           mh_recruit_df['subcontig_total']
    mh_recruit_df.sort_values(by='percent_recruited', ascending=False, inplace=True)
    # Only pass contigs that have the magjority of subcontigs recruited (>= 51%)
    mh_recruit_filter_df = mh_recruit_df.loc[mh_recruit_df['percent_recruited'] >=
                                                 float(mh_per_pass)
                                                 ]
    mg_contig_per_max_df = mh_recruit_filter_df.groupby(['contig_id'])[
        'percent_recruited'].max().reset_index()
    mg_contig_per_max_df.columns = ['contig_id', 'percent_max']
    mh_recruit_max_df = mh_recruit_filter_df.merge(mg_contig_per_max_df, how='left',
                                                       on='contig_id')
    # Now pass contigs that have the maximum recruit % of subcontigs
    mh_max_only_df = mh_recruit_max_df.loc[mh_recruit_max_df['percent_recruited'] >=
                                               mh_recruit_max_df['percent_max']
                                               ]
    mh_max_df = minhash_df[minhash_df['contig_id'].isin(list(mh_max_only_df['contig_id']))]

    mh_max_df.to_csv(o_join(mhr_path, mg_id + '.mhr_trimmed_recruits.tsv'), sep='\t',
                        index=False
                        )
    logging.info('[SABer]: MinHash Recruitment Algorithm Complete\n')

    return mh_max_df
