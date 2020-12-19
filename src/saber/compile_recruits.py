import matplotlib

matplotlib.use('agg')
from os.path import join as o_join
from os.path import basename
import pandas as pd
import saber.utilities as s_utils


def run_combine_recruits(final_path, ext_path, asm_path, check_path, mg_file, tetra_df_dict,
                         minhash_df, sag_list
                         ):
    mg_contigs_dict = s_utils.get_seqs(mg_file)
    mg_contigs = tuple([(r.name, r.seq) for r in mg_contigs_dict])

    for tetra_id in tetra_df_dict:
        if tetra_id == 'comb':
            tetra_df = tetra_df_dict[tetra_id]
            sag2path_dict = {}
            for sag_path in sag_list:
                base = basename(sag_path)
                sag_id = base.rsplit('.', 1)[0]
                sag2path_dict[sag_id] = sag_path

            # Merge MinHash and GMM Tetra (passed first by ABR)
            mh_gmm_merge_df = minhash_df[['sag_id', 'contig_id']].merge(
                tetra_df[['sag_id', 'contig_id']], how='outer',
                on=['sag_id', 'contig_id']
            ).drop_duplicates()

            mh_gmm_merge_df.to_csv(o_join(final_path, tetra_id + '.xPG.tsv'), sep='\t', index=True)
            mg_contigs_df = pd.DataFrame(mg_contigs, columns=['contig_id', 'seq'])
            sag_de_df_list = []
            for sag_id in set(mh_gmm_merge_df['sag_id']):
                final_rec = o_join(final_path, sag_id + '.' + tetra_id + '.xPG.fasta')

                sub_merge_df = mh_gmm_merge_df.loc[mh_gmm_merge_df['sag_id'] == sag_id]
                print('[SABer]: Recruited %s contigs from entire analysis for %s' %
                      (sub_merge_df.shape[0], sag_id)
                      )
                with open(o_join(final_path, sag_id + '.' + tetra_id + '.xPG.fasta'), 'w') as final_out:
                    mg_sub_filter_df = mg_contigs_df.loc[mg_contigs_df['contig_id'
                    ].isin(sub_merge_df['contig_id'])
                    ]
                    final_mgsubs_list = ['\n'.join(['>' + x[0], x[1]]) for x in
                                         zip(mg_sub_filter_df['contig_id'],
                                             mg_sub_filter_df['seq']
                                             )
                                         ]
                    final_out.write('\n'.join(final_mgsubs_list))
