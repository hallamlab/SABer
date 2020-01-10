import logging
import sourmash
from os.path import isfile
from os.path import join as o_join
import pandas as pd

def run_minhash_recruiter(sig_path, mhr_path, sag_subcontigs, mg_subcontigs, jacc_threshold=0.95):
    logging.info('[SABer]: Starting MinHash Recruitment Algorithm\n')
    # Calculate/Load MinHash Signatures with SourMash for MG subseqs
    mg_id, mg_headers, mg_subs = mg_subcontigs
    if isfile(o_join(sig_path, mg_id + '.metaG.sig')):  # TODO: MG should only be loaded if required
        logging.info('[SABer]: Loading %s Signatures' % mg_id)
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
                print('[SABer]: Loading Signature for %s\n' % sag_id)
                sag_sig = sourmash.signature.load_one_signature(o_join(sig_path,
                                                                     sag_id + '.SAG.sig')
                                                                )
            else:
                print('[SABer]: Building Signature for %s\n' % sag_id)
                sag_minhash = sourmash.MinHash(n=0, ksize=51, scaled=100)
                for sag_head, sag_subseq in zip(sag_headers, sag_subs):
                    sag_upseq = sag_subseq.upper()
                    sag_minhash.add_sequence(sag_upseq, force=True)
                sag_sig = sourmash.SourmashSignature(sag_minhash, name=sag_id)
                with open(o_join(sig_path, sag_id + '.SAG.sig'), 'w') as sags_out:
                    sourmash.signature.save_signatures([sag_sig], fp=sags_out)
            print('[SABer]: Comparing  %s and MetaG signatures\n' % sag_id)
            pass_list = []
            mg_sig_list = list(mg_sig_list)
            for j, mg_sig in enumerate(mg_sig_list):
                jacc_sim = mg_sig.contained_by(sag_sig)
                mg_nm = mg_sig.name()
                if jacc_sim >= jacc_threshold:
                    pass_list.append([sag_id, mg_nm, mg_nm.rsplit('_', 1)[0]])

            with open(o_join(mhr_path, sag_id + '.mhr_recruits.tsv'), 'w') as mhr_out:
                mhr_out.write('\n'.join(['\t'.join(x) for x in pass_list]))
        minhash_pass_list.extend(pass_list)
        print('[SABer]: MinHash recruited %s subcontigs to %s' % (str(len(minhash_pass_list)), sag_id))

    minhash_df = pd.DataFrame(minhash_pass_list, columns=['sag_id', 'subcontig_id',
                                                          'contig_id'
                                                          ])
    return minhash_df
