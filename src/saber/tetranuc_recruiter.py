import argparse
import logging
import multiprocessing
import sys
from os.path import isfile, basename
from os.path import join as o_join

import numpy as np
import pandas as pd
import saber.logger as s_log
import saber.utilities as s_utils
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler


def run_tetra_recruiter(tra_path, sag_sub_files, mg_sub_file, rpkm_max_df, minhash_df,
                        per_pass, nthreads, force
                        ):
    """Returns dataframe of subcontigs recruited via tetranucleotide Hz

    Parameters:
    tra_path (str): string of global path to tetrenucleotide output directory
    sag_sub_files (list): list containing sublists with two values: [sag_id, sag_path]
                          where sag_id (str) is a unique ID for a SAG and sag_path is
                          the global path the the SAG subcontig fasta file
    mg_sub_file (list): list containing two values: mg_id and mg_file. (same as sag_sub_files)
    rpkm_max_df (df): dataframe containing the abundance recruits from the previous step.
    per_pass (float): percent of agreement for subcontig classification to pass the complete
                          contig (default is 0.01)

    """
    # TODO: 1. Think about using Minimum Description Length (MDL) instead of AIC/BIC
    #        2. [Normalized Maximum Likelihood or Fish Information Approximation]
    #        3. Can TetraNuc Hz be calc'ed for each sample? Does that improve things?
    #            (think about http://merenlab.org/2020/01/02/visualizing-metagenomic-bins/#introduction)

    logging.info('[SABer]: Starting Tetranucleotide Recruitment Step\n')

    mg_id = mg_sub_file[0]
    # Build/Load tetramers for SAGs and MG subset by ara recruits
    if isfile(o_join(tra_path, mg_id + '.tetras.tsv')):
        logging.info('[SABer]: Loading tetramer Hz matrix for %s\n' % mg_id)
        mg_tetra_df = pd.read_csv(o_join(tra_path, mg_id + '.tetras.tsv'),
                                  sep='\t', index_col=0, header=0
                                  )
        mg_headers = mg_tetra_df.index.values
    else:
        logging.info('[SABer]: Calculating tetramer Hz matrix for %s\n' % mg_id)
        mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])  # TODO: can this be removed?
        mg_headers = tuple(mg_subcontigs.keys())
        mg_subs = tuple([r.seq for r in mg_subcontigs])
        mg_tetra_df = s_utils.tetra_cnt(mg_subs)
        mg_tetra_df['contig_id'] = mg_headers
        mg_tetra_df.set_index('contig_id', inplace=True)
        mg_tetra_df.to_csv(o_join(tra_path, mg_id + '.tetras.tsv'),
                           sep='\t'
                           )
    ####
    gmm_df_list = []
    svm_df_list = []
    iso_df_list = []
    comb_df_list = []
    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for sag_id in set(minhash_df['sag_id']):
        arg_list.append([sag_id, mg_id, mg_headers, tra_path, minhash_df, mg_tetra_df, rpkm_max_df, force])
    results = pool.imap_unordered(run_tetra_ML, arg_list)
    for i, output in enumerate(results):
        gmm_df_list.append(output[0])
        svm_df_list.append(output[1])
        iso_df_list.append(output[2])
        comb_df_list.append(output[3])
        sys.stderr.write('\rdone {}/{}'.format(i, len(arg_list)))
    pool.close()
    pool.join()
    ####
    gmm_concat_df = pd.concat(gmm_df_list)
    svm_concat_df = pd.concat(svm_df_list)
    iso_concat_df = pd.concat(iso_df_list)
    comb_concat_df = pd.concat(comb_df_list)
    gmm_concat_df.to_csv(o_join(tra_path, mg_id + '.gmm.tra_trimmed_recruits.tsv'), sep='\t',
                         index=False
                         )
    svm_concat_df.to_csv(o_join(tra_path, mg_id + '.svm.tra_trimmed_recruits.tsv'), sep='\t',
                         index=False
                         )
    iso_concat_df.to_csv(o_join(tra_path, mg_id + '.iso.tra_trimmed_recruits.tsv'), sep='\t',
                         index=False
                         )
    comb_concat_df.to_csv(o_join(tra_path, mg_id + '.comb.tra_trimmed_recruits.tsv'), sep='\t',
                          index=False
                          )

    tetra_df_dict = {'gmm': gmm_concat_df, 'svm': svm_concat_df, 'iso': iso_concat_df,
                     'comb': comb_concat_df
                     }

    return tetra_df_dict


def run_tetra_ML(p):
    sag_id, mg_id, mg_headers, tra_path, minhash_df, mg_tetra_df, rpkm_max_df, force = p
    # sag_id, sag_file = sag_rec
    if sag_id in list(rpkm_max_df['sag_id']):

        if (isfile(o_join(tra_path, sag_id + '.gmm_recruits.tsv')) &
                isfile(o_join(tra_path, sag_id + '.svm_recruits.tsv')) &
                isfile(o_join(tra_path, sag_id + '.iso_recruits.tsv')) &
                isfile(o_join(tra_path, sag_id + '.comb_recruits.tsv')) &
                force == False
        ):
            logging.info('[SABer]: Loading  %s tetramer Hz recruit list\n' % sag_id)
            # with open(o_join(tra_path, sag_id + '.gmm_recruits.tsv'), 'r') as tra_in:
            #    gmm_pass_list = [x.rstrip('\n').split('\t') for x in tra_in.readlines()]
            # with open(o_join(tra_path, sag_id + '.svm_recruits.tsv'), 'r') as tra_in:
            #    svm_pass_list = [x.rstrip('\n').split('\t') for x in tra_in.readlines()]
            # with open(o_join(tra_path, sag_id + '.iso_recruits.tsv'), 'r') as tra_in:
            #    iso_pass_list = [x.rstrip('\n').split('\t') for x in tra_in.readlines()]
            # with open(o_join(tra_path, sag_id + '.comb_recruits.tsv'), 'r') as tra_in:
            #    comb_pass_list = [x.rstrip('\n').split('\t') for x in tra_in.readlines()]
            gmm_filter_df = pd.read_csv(o_join(tra_path, sag_id + '.gmm_recruits.tsv'),
                                        sep='\t', header=0
                                        )
            svm_filter_df = pd.read_csv(o_join(tra_path, sag_id + '.svm_recruits.tsv'),
                                        sep='\t', header=0
                                        )
            iso_filter_df = pd.read_csv(o_join(tra_path, sag_id + '.iso_recruits.tsv'),
                                        sep='\t', header=0
                                        )
            comb_filter_df = pd.read_csv(o_join(tra_path, sag_id + '.comb_recruits.tsv'),
                                         sep='\t', header=0
                                         )

        else:
            # Concat SAGs amd MG for ML Training
            minhash_sag_df = minhash_df.loc[(minhash_df['sag_id'] == sag_id)]  # &
            # (minhash_df['jacc_sim'] == 1.0)
            # ]
            if minhash_sag_df.shape[0] != 0:

                scale = StandardScaler().fit(mg_tetra_df.values)
                scaled_data = scale.transform(mg_tetra_df.values)
                std_tetra_df = pd.DataFrame(scaled_data, index=mg_tetra_df.index)

                sag_mh_contig_list = list(set(minhash_sag_df['contig_id'].values))
                sag_tetra_contig_list = [x for x in std_tetra_df.index.values
                                         if sag_mh_contig_list.count(x.rsplit('_', 1)[0]) != 0
                                         ]
                sag_tetra_df = std_tetra_df.loc[std_tetra_df.index.isin(sag_tetra_contig_list)]
                mg_rpkm_contig_list = list(set(rpkm_max_df.loc[rpkm_max_df['sag_id'] == sag_id
                                                               ]['contig_id'].values)
                                           )
                mg_tetra_contig_list = [x for x in std_tetra_df.index.values
                                        if mg_rpkm_contig_list.count(x.rsplit('_', 1)[0]) != 0
                                        ]
                mg_tetra_filter_df = std_tetra_df.loc[std_tetra_df.index.isin(mg_tetra_contig_list)]

                # logging.info('[SABer]: Calculating AIC/BIC for GMM components\n')
                sag_train_vals = [1 for x in sag_tetra_df.index]
                n_components = np.arange(1, 100, 1)
                models = [GMM(n, random_state=42) for n in n_components]
                bics = []
                aics = []
                min_bic = None
                min_aic = None
                # bic_counter = 0
                aic_counter = 0
                for i, model in enumerate(models):
                    n_comp = n_components[i]
                    '''
                    if bic_counter <= 10:
                        try:
                            bic = model.fit(sag_tetra_df.values,
                                            sag_train_vals).bic(sag_tetra_df.values
                                                                )
                            bics.append(bic)
                        except:
                            1+1
                            #logging.info('[WARNING]: BIC failed with %s components\n' % n_comp)
                        if min_bic is None:
                            min_bic = bic
                        elif min_bic > bic:
                            min_bic = bic
                            bic_counter = 0
                        else:
                            bic_counter += 1
                    '''
                    if aic_counter <= 10:
                        try:
                            aic = model.fit(sag_tetra_df.values,
                                            sag_train_vals).aic(sag_tetra_df.values
                                                                )
                            aics.append(aic)
                        except:
                            1 + 1
                            # logging.info('[WARNING]: AIC failed with %s components\n' % n_comp)
                        if min_aic is None:
                            min_aic = aic
                        elif min_aic > aic:
                            min_aic = aic
                            aic_counter = 0
                        else:
                            aic_counter += 1

                # min_bic_comp = n_components[bics.index(min_bic)]
                min_aic_comp = n_components[aics.index(min_aic)]
                # logging.info('[SABer]: Min AIC/BIC at %s/%s, respectively\n' %
                #      (min_aic_comp, min_bic_comp)
                #      )
                # logging.info('[SABer]: Using BIC as guide for GMM components\n')
                # logging.info('[SABer]: Training GMM on SAG tetras\n')
                gmm = GMM(n_components=min_aic_comp, random_state=42
                          ).fit(sag_tetra_df.values)
                # logging.info('[SABer]: GMM Converged: %s\n' % gmm.converged_)
                try:  # TODO: add predict and predict_proba to this and output all to table
                    sag_scores = gmm.score_samples(sag_tetra_df.values)
                    sag_scores_df = pd.DataFrame(data=sag_scores, index=sag_tetra_df.index.values)
                    sag_scores_df.columns = ['wLogProb']
                    sag_score_min = min(sag_scores_df.values)[0]
                    sag_score_max = max(sag_scores_df.values)[0]
                    mg_scores = gmm.score_samples(mg_tetra_filter_df.values)
                    mg_scores_df = pd.DataFrame(data=mg_scores, index=mg_tetra_filter_df.index.values)
                    mg_scores_df.columns = ['wLogProb']
                    gmm_pass_df = mg_scores_df.loc[(mg_scores_df['wLogProb'] >= sag_score_min) &
                                                   (mg_scores_df['wLogProb'] <= sag_score_max)
                                                   ]
                    # And is has to be from the RPKM pass list
                    # gmm_pass_df = gmm_pass_df.loc[gmm_pass_df.index.isin(mg_rpkm_contig_list)]
                    gmm_pass_list = []
                    for md_nm in gmm_pass_df.index.values:
                        if mg_tetra_contig_list.count(md_nm) != 0:
                            gmm_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
                except:
                    # logging.info('[SABer]: Warning: No recruits found...\n')
                    gmm_pass_list = []

                # logging.info('[SABer]: Training OCSVM on SAG tetras\n')
                # fit OCSVM
                clf = svm.OneClassSVM(nu=0.9, gamma=0.0001)
                clf.fit(sag_tetra_df.values)
                # print(clf.get_params())
                sag_pred = clf.predict(sag_tetra_df.values)
                # sag_pred_df = pd.DataFrame(data=sag_pred, index=sag_tetra_df.index.values)
                mg_pred = clf.predict(mg_tetra_filter_df.values)
                mg_pred_df = pd.DataFrame(data=mg_pred, index=mg_tetra_filter_df.index.values)
                svm_pass_df = mg_pred_df.loc[mg_pred_df[0] != -1]
                # And is has to be from the RPKM pass list
                # svm_pass_df = svm_pass_df.loc[svm_pass_df.index.isin(mg_rpkm_contig_list)]
                svm_pass_list = []
                for md_nm in svm_pass_df.index.values:
                    if mg_tetra_contig_list.count(md_nm) != 0:
                        svm_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])

                # logging.info('[SABer]: Training Isolation Forest on SAG tetras\n')
                # fit IsoForest
                clf = IsolationForest(random_state=42)
                clf.fit(sag_tetra_df.values)
                # clf.set_params(n_estimators=20)  # add 10 more trees
                # clf.fit(sag_tetra_df.values)  # fit the added trees
                mg_pred = clf.predict(mg_tetra_filter_df.values)
                mg_pred_df = pd.DataFrame(data=mg_pred, index=mg_tetra_filter_df.index.values)
                iso_pass_df = mg_pred_df.loc[mg_pred_df[0] != -1]
                # And is has to be from the RPKM pass list
                # iso_pass_df = iso_pass_df.loc[iso_pass_df.index.isin(mg_rpkm_contig_list)]
                iso_pass_list = []
                for md_nm in iso_pass_df.index.values:
                    if mg_tetra_contig_list.count(md_nm) != 0:
                        iso_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])

                gmm_df = pd.DataFrame(gmm_pass_list, columns=['sag_id', 'subcontig_id', 'contig_id'])
                svm_df = pd.DataFrame(svm_pass_list, columns=['sag_id', 'subcontig_id', 'contig_id'])
                iso_df = pd.DataFrame(iso_pass_list, columns=['sag_id', 'subcontig_id', 'contig_id'])

                gmm_filter_df = filter_tetras(sag_id, mg_headers, 'gmm', gmm_df)
                svm_filter_df = filter_tetras(sag_id, mg_headers, 'svm', svm_df)
                iso_filter_df = filter_tetras(sag_id, mg_headers, 'iso', iso_df)

                gmm_id_list = list(gmm_filter_df['subcontig_id'])
                svm_id_list = list(svm_filter_df['subcontig_id'])
                iso_id_list = list(iso_filter_df['subcontig_id'])

                ab_set = set(gmm_id_list).intersection(svm_id_list)
                ac_set = set(gmm_id_list).intersection(iso_id_list)
                bc_set = set(svm_id_list).intersection(iso_id_list)
                # comb_set_list = list({*ab_set, *ac_set, *bc_set})
                comb_set_list = list(set(list(ab_set) + list(ac_set) + list(bc_set)))
                # comb_set_list = list(set(list(gmm_id_list) + list(svm_id_list) + list(iso_id_list)))
                # comb_set_list = list(ac_set)
                comb_pass_list = []
                for md_nm in comb_set_list:
                    comb_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
                comb_df = pd.DataFrame(comb_pass_list,
                                       columns=['sag_id', 'subcontig_id', 'contig_id']
                                       )
                comb_filter_df = filter_tetras(sag_id, mg_headers, 'comb', comb_df)

                gmm_filter_df.to_csv(o_join(tra_path, sag_id + '.gmm_recruits.tsv'),
                                     sep='\t', index=False
                                     )
                svm_filter_df.to_csv(o_join(tra_path, sag_id + '.svm_recruits.tsv'),
                                     sep='\t', index=False
                                     )
                iso_filter_df.to_csv(o_join(tra_path, sag_id + '.iso_recruits.tsv'),
                                     sep='\t', index=False
                                     )
                comb_filter_df.to_csv(o_join(tra_path, sag_id + '.comb_recruits.tsv'),
                                      sep='\t', index=False
                                      )
            else:
                gmm_pass_list = []
                svm_pass_list = []
                iso_pass_list = []
                comb_pass_list = []
                gmm_filter_df = pd.DataFrame(gmm_pass_list,
                                             columns=['sag_id', 'subcontig_id', 'contig_id']
                                             )
                svm_filter_df = pd.DataFrame(svm_pass_list,
                                             columns=['sag_id', 'subcontig_id', 'contig_id']
                                             )
                iso_filter_df = pd.DataFrame(iso_pass_list,
                                             columns=['sag_id', 'subcontig_id', 'contig_id']
                                             )
                comb_filter_df = pd.DataFrame(comb_pass_list,
                                              columns=['sag_id', 'subcontig_id', 'contig_id']
                                              )

        return gmm_filter_df, svm_filter_df, iso_filter_df, comb_filter_df

    else:
        gmm_pass_list = []
        svm_pass_list = []
        iso_pass_list = []
        comb_pass_list = []
        gmm_filter_df = pd.DataFrame(gmm_pass_list,
                                     columns=['sag_id', 'subcontig_id', 'contig_id']
                                     )
        svm_filter_df = pd.DataFrame(svm_pass_list,
                                     columns=['sag_id', 'subcontig_id', 'contig_id']
                                     )
        iso_filter_df = pd.DataFrame(iso_pass_list,
                                     columns=['sag_id', 'subcontig_id', 'contig_id']
                                     )
        comb_filter_df = pd.DataFrame(comb_pass_list,
                                      columns=['sag_id', 'subcontig_id', 'contig_id']
                                      )
        return gmm_filter_df, svm_filter_df, iso_filter_df, comb_filter_df


def filter_tetras(sag_id, mg_headers, tetra_id, tetra_df):
    # tetra_df = tetra_df_dict[tetra_id]
    # Count # of subcontigs recruited to each SAG
    cnt_df = tetra_df.groupby(['sag_id', 'contig_id']).count().reset_index()
    cnt_df.columns = ['sag_id', 'contig_id', 'subcontig_recruits']
    # Build subcontig count for each MG contig
    mg_contig_list = [x.rsplit('_', 1)[0] for x in mg_headers]
    mg_tot_df = pd.DataFrame(zip(mg_contig_list, mg_headers),
                             columns=['contig_id', 'subcontig_id'])
    mg_tot_cnt_df = mg_tot_df.groupby(['contig_id']).count().reset_index()
    mg_tot_cnt_df.columns = ['contig_id', 'subcontig_total']
    mg_recruit_df = cnt_df.merge(mg_tot_cnt_df, how='left', on='contig_id')
    mg_recruit_df['percent_recruited'] = mg_recruit_df['subcontig_recruits'] / \
                                         mg_recruit_df['subcontig_total']
    mg_recruit_df.sort_values(by='percent_recruited', ascending=False, inplace=True)
    # Only pass contigs that have the magjority of subcontigs recruited (>= N%)
    if ((tetra_id == 'svm') or (tetra_id == 'gmm')):
        mg_recruit_filter_df = mg_recruit_df.loc[
            # mg_recruit_df['percent_recruited'] >= 0.51
            mg_recruit_df['subcontig_recruits'] >= 3
            ]
    elif ((tetra_id == 'comb') or (tetra_id == 'iso')):
        mg_recruit_filter_df = mg_recruit_df.loc[
            mg_recruit_df['percent_recruited'] >= 0.51
            ]
    tetra_max_list = []
    sag_max_only_df = mg_recruit_filter_df.loc[
        mg_recruit_filter_df['sag_id'] == sag_id
        ]
    tetra_max_df = mg_tot_df[mg_tot_df['contig_id'].isin(
        list(sag_max_only_df['contig_id'])
    )]
    tetra_max_df['sag_id'] = sag_id
    tetra_max_df = tetra_max_df[['sag_id', 'subcontig_id', 'contig_id']]

    return tetra_max_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='uses tetrenucleotide Hz to recruit metaG reads to SAGs')
    parser.add_argument(
        '--tetra_path', help='path to tetrenucleotide output directory',
        required=True
    )
    parser.add_argument(
        '--sag_sub_file',
        help='path to SAG subcontigs file', required=True
    )
    parser.add_argument(
        '--mg_sub_file',
        help='path to metagenome subcontigs file', required=True
    )
    parser.add_argument(
        '--abund_df',
        help='path to output dataframe from abundance recruiter', required=True
    )
    parser.add_argument(
        '--per_pass',
        help='pass percentage of subcontigs to pass complete contig', required=True,
        default='0.01'
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Prints a more verbose runtime log"
                        )
    args = parser.parse_args()
    # set args
    tra_path = args.tetra_path
    sag_sub_file = args.sag_sub_file
    mg_sub_file = args.mg_sub_file
    abund_recruit_file = args.abund_df
    per_pass = float(args.per_pass)

    s_log.prep_logging("tetra_log.txt", args.verbose)
    sag_id = basename(sag_sub_file).rsplit('.', 2)[0]
    mg_id = basename(mg_sub_file).rsplit('.', 2)[0]
    abund_recruit_df = pd.read_csv(abund_recruit_file, header=0, sep='\t')
    logging.info('[SABer]: Starting Tetranucleotide Recruitment Step\n')
    run_tetra_recruiter(tra_path, [[sag_id, sag_sub_file]], [mg_id, mg_sub_file],
                        abund_recruit_df, per_pass)
