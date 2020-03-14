import logging
import sourmash
from os.path import isfile
from os.path import join as o_join
import pandas as pd

def run_minhash_recruiter(sig_path, mhr_path, sag_subcontigs, mg_subcontigs,
                            jacc_threshold, mh_per_pass
                            ):
    logging.info('[SABer]: Starting MinHash Recruitment Algorithm\n')
    # Calculate/Load MinHash Signatures with SourMash for MG subseqs
    mg_id, mg_headers, mg_subs = mg_subcontigs
    if isfile(o_join(sig_path, mg_id + '.metaG.sig')):  # TODO: MG should only be loaded if required
        logging.info('[SABer]: Loading %s Signatures\n' % mg_id)
        mg_sig_list = sourmash.signature.load_signatures(o_join(sig_path, mg_id + \
                                                                      '.metaG.sig')
                                                                        )
    else:
        logging.info('[SABer]: Building Signatures for %s\n' % mg_id)
        mg_sig_list = []
        for mg_head, seq in zip(mg_headers, mg_subs):
            up_seq = seq.upper()
            mg_minhash = sourmash.MinHash(n=0, ksize=51, scaled=100)
            mg_minhash.add_sequence(up_seq, force=True)
            mg_sig = sourmash.SourmashSignature(mg_minhash, name=mg_head)
            mg_sig_list.append(mg_sig)
        with open(o_join(sig_path, mg_id + '.metaG.sig'), 'w') as mg_out:
            sourmash.signature.save_signatures(mg_sig_list, fp=mg_out)

    # Load comparisons OR Compare SAG sigs to MG sigs to find containment
    logging.info('[SABer]: Comparing Signatures of SAGs to MetaG contigs\n')
    minhash_pass_list = []
    for sag_id, sag_headers, sag_subs in sag_subcontigs:
        if isfile(o_join(mhr_path, sag_id + '.mhr_recruits.tsv')):
            logging.info('[SABer]: Loading %s and MetaG signature recruit list\n' % sag_id)
            with open(o_join(mhr_path, sag_id + '.mhr_recruits.tsv'), 'r') as mhr_in:
                pass_list = [x.rstrip('\n').split('\t') for x in mhr_in.readlines()]
        else:
            # Calculate\Load MinHash Signatures with SourMash for SAG subseqs
            if isfile(o_join(sig_path, sag_id + '.SAG.sig')):
                logging.info('[SABer]: Loading Signature for %s\n' % sag_id)
                sag_sig = sourmash.signature.load_one_signature(o_join(sig_path,
                                                                     sag_id + '.SAG.sig')
                                                                )
            else:
                logging.info('[SABer]: Building Signature for %s\n' % sag_id)
                sag_minhash = sourmash.MinHash(n=0, ksize=51, scaled=100)
                for sag_head, sag_subseq in zip(sag_headers, sag_subs):
                    sag_upseq = sag_subseq.upper()
                    sag_minhash.add_sequence(sag_upseq, force=True)
                sag_sig = sourmash.SourmashSignature(sag_minhash, name=sag_id)
                with open(o_join(sig_path, sag_id + '.SAG.sig'), 'w') as sags_out:
                    sourmash.signature.save_signatures([sag_sig], fp=sags_out)
            logging.info('[SABer]: Comparing  %s and MetaG signature\n' % sag_id)
            pass_list = []
            mg_sig_list = list(mg_sig_list)
            for mg_sig in mg_sig_list:
                jacc_sim = mg_sig.contained_by(sag_sig)
                mg_nm = mg_sig.name()
                if jacc_sim >= jacc_threshold:
                    pass_list.append([sag_id, mg_nm, mg_nm.rsplit('_', 1)[0]])

            with open(o_join(mhr_path, sag_id + '.mhr_recruits.tsv'), 'w') as mhr_out:
                mhr_out.write('\n'.join(['\t'.join(x) for x in pass_list]))
        minhash_pass_list.extend(pass_list)
        logging.info('[SABer]: MinHash recruited %s subcontigs to %s\n'
                        % (str(len(pass_list)), sag_id)
                        )

    minhash_df = pd.DataFrame(minhash_pass_list, columns=['sag_id', 'subcontig_id',
                                                          'contig_id'
                                                          ])

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

    return mh_max_df
