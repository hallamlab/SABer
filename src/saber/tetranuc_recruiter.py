import logging
import pandas as pd
import numpy as np
from os.path import isfile
from os.path import join as o_join
import saber.utilities as s_utils
from sklearn.mixture import GaussianMixture as GMM
from sklearn import svm
from sklearn.ensemble import IsolationForest


def run_tetra_recruiter(tra_path, sag_sub_files, mg_sub_file, rpkm_max_df, gmm_per_pass):
    # TODO: 1. Think about using Minimum Description Length (MDL) instead of AIC/BIC
    #        2. [Normalized Maximum Likelihood or Fish Information Approximation]
    #        3. Can TetraNuc Hz be calc'ed for each sample? Does that improve things?
    #            (think about http://merenlab.org/2020/01/02/visualizing-metagenomic-bins/#introduction)

    mg_id = mg_sub_file[0]
    mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
    mg_headers = tuple(mg_subcontigs.keys())

    mg_subs = tuple([r.seq for r in mg_subcontigs])
    #mg_id, mg_headers, mg_subs = mg_subcontigs

    # Build/Load tetramers for SAGs and MG subset by ara recruits
    if isfile(o_join(tra_path, mg_id + '.tetras.tsv')):
        logging.info('[SABer]: Loading tetramer Hz matrix for %s\n' % mg_id)
        mg_tetra_df = pd.read_csv(o_join(tra_path, mg_id + '.tetras.tsv'),
                                  sep='\t', index_col=0, header=0
                                  )
    else:
        logging.info('[SABer]: Calculating tetramer Hz matrix for %s\n' % mg_id)
        mg_tetra_df = pd.DataFrame.from_dict(s_utils.tetra_cnt(mg_subs))
        mg_tetra_df['contig_id'] = mg_headers
        mg_tetra_df.set_index('contig_id', inplace=True)
        mg_tetra_df.to_csv(o_join(tra_path, mg_id + '.tetras.tsv'),
                           sep='\t'
                           )

    gmm_total_pass_list = []
    svm_total_pass_list = []
    iso_total_pass_list= []
    comb_total_pass_list = []
    #for sag_id, sag_headers, sag_subs in sag_subcontigs:
    for i, sag_rec in enumerate(sag_sub_files):
        sag_id, sag_file = sag_rec
        sag_subcontigs = s_utils.get_seqs(sag_file)
        sag_headers= tuple(sag_subcontigs.keys())
        sag_subs = tuple([r.seq for r in sag_subcontigs.itervalues()])

        if (isfile(o_join(tra_path, sag_id + '.gmm_recruits.tsv')) &
            isfile(o_join(tra_path, sag_id + '.svm_recruits.tsv')) &
            isfile(o_join(tra_path, sag_id + '.iso_recruits.tsv')) &
            isfile(o_join(tra_path, sag_id + '.comb_recruits.tsv'))
            ):
            logging.info('[SABer]: Loading  %s tetramer Hz recruit list\n' % sag_id)
            with open(o_join(tra_path, sag_id + '.gmm_recruits.tsv'), 'r') as tra_in:
                gmm_pass_list = [x.rstrip('\n').split('\t') for x in tra_in.readlines()]
            with open(o_join(tra_path, sag_id + '.svm_recruits.tsv'), 'r') as tra_in:
                svm_pass_list = [x.rstrip('\n').split('\t') for x in tra_in.readlines()]
            with open(o_join(tra_path, sag_id + '.iso_recruits.tsv'), 'r') as tra_in:
                iso_pass_list = [x.rstrip('\n').split('\t') for x in tra_in.readlines()]
            with open(o_join(tra_path, sag_id + '.comb_recruits.tsv'), 'r') as tra_in:
                comb_pass_list = [x.rstrip('\n').split('\t') for x in tra_in.readlines()]
        else:
            if isfile(o_join(tra_path, sag_id + '.tetras.tsv')):
                logging.info('[SABer]: Loading tetramer Hz matrix for %s\n' % sag_id)
                sag_tetra_df = pd.read_csv(o_join(tra_path, sag_id + '.tetras.tsv'),
                                           sep='\t', index_col=0, header=0)
            else:
                logging.info('[SABer]: Calculating tetramer Hz matrix for %s\n' % sag_id)
                sag_tetra_df = pd.DataFrame.from_dict(s_utils.tetra_cnt(sag_subs))
                sag_tetra_df['contig_id'] = sag_headers
                sag_tetra_df.set_index('contig_id', inplace=True)
                sag_tetra_df.to_csv(o_join(tra_path, sag_id + '.tetras.tsv'), sep='\t')

            # Concat SAGs amd MG for GMM
            mg_rpkm_contig_list = list(rpkm_max_df.loc[rpkm_max_df['sag_id'] == sag_id
                                                       ]['subcontig_id'].values
                                       )

            mg_tetra_filter_df = mg_tetra_df.loc[mg_tetra_df.index.isin(mg_rpkm_contig_list)]
            #concat_tetra_df = pd.concat([sag_tetra_df, mg_tetra_filter_df])
            '''
            normed_tetra_df = concat_tetra_df
            sag_normed_tetra_df = normed_tetra_df[
                normed_tetra_df.index.isin(sag_tetra_df.index)
            ]
            mg_normed_tetra_df = normed_tetra_df.loc[
                normed_tetra_df.index.isin(mg_tetra_filter_df.index)
            ]

            # UMAP for Dimension reduction of tetras
            sag_features = sag_normed_tetra_df.values
            sag_targets = sag_normed_tetra_df.index.values
            mg_features = mg_normed_tetra_df.values
            mg_targets = mg_normed_tetra_df.index.values
            normed_features = normed_tetra_df.values
            normed_targets = normed_tetra_df.index.values

            logging.info('[SABer]: Dimension reduction of tetras with UMAP\n')
            umap_trans = umap.UMAP(n_neighbors=2, min_dist=0.0,
                                   n_components=num_components, metric='manhattan',
                                   random_state=42
                                   ).fit_transform(normed_features)
            pc_col_names = ['pc' + str(x) for x in range(1, num_components + 1)]
            umap_df = pd.DataFrame(umap_trans, columns=pc_col_names, index=normed_targets)

            sag_umap_df = umap_df.loc[umap_df.index.isin(sag_tetra_df.index)]
            mg_umap_df = umap_df.loc[umap_df.index.isin(mg_tetra_filter_df.index)]

            sag_tetra_df = concat_tetra_df.loc[
                                    concat_tetra_df.index.isin(sag_tetra_df.index)
                                    ]
            mg_tetra_df = concat_tetra_df.loc[
                                    concat_tetra_df.index.isin(mg_tetra_filter_df.index)
                                    ]
            '''

            logging.info('[SABer]: Calculating AIC/BIC for GMM components\n')
            sag_train_vals = [1 for x in sag_tetra_df.index]
            n_components = np.arange(1, 5, 1)
            models = [GMM(n, random_state=42) for n in n_components]
            bics = []
            aics = []
            for i, model in enumerate(models):
                n_comp = n_components[i]
                try:
                    bic = model.fit(sag_tetra_df.values,
                                    sag_train_vals).bic(sag_tetra_df.values
                                                        )
                    bics.append(bic)
                except:
                    logging.info('[WARNING]: BIC failed with %s components\n' % n_comp)
                try:
                    aic = model.fit(sag_tetra_df.values,
                                    sag_train_vals).aic(sag_tetra_df.values
                                                        )
                    aics.append(aic)
                except:
                    logging.info('[WARNING]: AIC failed with %s components\n' % n_comp)

            min_bic_comp = n_components[bics.index(min(bics))]
            min_aic_comp = n_components[aics.index(min(aics))]
            logging.info('[SABer]: Min AIC/BIC at %s/%s, respectively\n' %
                  (min_aic_comp, min_bic_comp)
                  )
            logging.info('[SABer]: Using BIC as guide for GMM components\n')
            logging.info('[SABer]: Training GMM on SAG tetras\n')
            gmm = GMM(n_components=min_bic_comp, random_state=42
                      ).fit(sag_tetra_df.values)
            logging.info('[SABer]: GMM Converged: %s\n' % gmm.converged_)
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
                gmm_pass_df = gmm_pass_df.loc[gmm_pass_df.index.isin(mg_rpkm_contig_list)]
                gmm_pass_list = []
                for md_nm in gmm_pass_df.index.values:
                    gmm_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
            except:
                logging.info('[SABer]: Warning: No recruits found...\n')
                gmm_pass_list = []


            logging.info('[SABer]: Training OCSVM on SAG tetras\n')
            # fit OCSVM
            clf = svm.OneClassSVM()
            clf.fit(sag_tetra_df.values)
            sag_pred = clf.predict(sag_tetra_df.values)
            #sag_pred_df = pd.DataFrame(data=sag_pred, index=sag_tetra_df.index.values)
            mg_pred = clf.predict(mg_tetra_filter_df.values)
            mg_pred_df = pd.DataFrame(data=mg_pred, index=mg_tetra_filter_df.index.values)
            svm_pass_df = mg_pred_df.loc[mg_pred_df[0] != -1]
            # And is has to be from the RPKM pass list
            svm_pass_df = svm_pass_df.loc[svm_pass_df.index.isin(mg_rpkm_contig_list)]
            svm_pass_list = []
            for md_nm in svm_pass_df.index.values:
                svm_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])


            logging.info('[SABer]: Training Isolation Forest on SAG tetras\n')
            # fit IsoForest
            clf = IsolationForest(random_state=42)
            clf.fit(sag_tetra_df.values)
            #clf.set_params(n_estimators=20)  # add 10 more trees
            #clf.fit(sag_tetra_df.values)  # fit the added trees
            mg_pred = clf.predict(mg_tetra_filter_df.values)
            mg_pred_df = pd.DataFrame(data=mg_pred, index=mg_tetra_filter_df.index.values)
            iso_pass_df = mg_pred_df.loc[mg_pred_df[0] != -1]
            # And is has to be from the RPKM pass list
            iso_pass_df = iso_pass_df.loc[iso_pass_df.index.isin(mg_rpkm_contig_list)]
            iso_pass_list = []
            for md_nm in iso_pass_df.index.values:
                iso_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])

            gmm_id_list = [x[1] for x in gmm_pass_list]
            svm_id_list = [x[1] for x in svm_pass_list]
            iso_id_list = [x[1] for x in iso_pass_list]
            comb_set_list = list(set(gmm_id_list) & set(svm_id_list) & set(iso_id_list))
            #comb_set_list = list(set(gmm_id_list) & set(svm_id_list))
            comb_pass_list = []
            for md_nm in comb_set_list:
                comb_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])

            logging.info('[SABer]: Recruited %s subcontigs to %s with GMM\n' % (len(gmm_pass_list), sag_id))
            logging.info('[SABer]: Recruited %s subcontigs to %s with SVM\n' % (len(svm_pass_list), sag_id))
            logging.info('[SABer]: Recruited %s subcontigs to %s with Isolation Forest\n' % (len(iso_pass_list), sag_id))
            logging.info('[SABer]: Recruited %s subcontigs to %s with combined methods\n' % (len(comb_pass_list), sag_id))

            with open(o_join(tra_path, sag_id + '.gmm_recruits.tsv'), 'w') as tra_out:
                tra_out.write('\n'.join(['\t'.join(x) for x in gmm_pass_list]))
            with open(o_join(tra_path, sag_id + '.svm_recruits.tsv'), 'w') as tra_out:
                tra_out.write('\n'.join(['\t'.join(x) for x in svm_pass_list]))
            with open(o_join(tra_path, sag_id + '.iso_recruits.tsv'), 'w') as tra_out:
                tra_out.write('\n'.join(['\t'.join(x) for x in iso_pass_list]))
            with open(o_join(tra_path, sag_id + '.comb_recruits.tsv'), 'w') as tra_out:
                tra_out.write('\n'.join(['\t'.join(x) for x in comb_pass_list]))

        gmm_total_pass_list.extend(gmm_pass_list)
        svm_total_pass_list.extend(svm_pass_list)
        iso_total_pass_list.extend(iso_pass_list)
        comb_total_pass_list.extend(comb_pass_list)

    gmm_df = pd.DataFrame(gmm_total_pass_list, columns=['sag_id', 'subcontig_id', 'contig_id'])
    svm_df = pd.DataFrame(svm_total_pass_list, columns=['sag_id', 'subcontig_id', 'contig_id'])
    iso_df = pd.DataFrame(iso_total_pass_list, columns=['sag_id', 'subcontig_id', 'contig_id'])
    comb_df = pd.DataFrame(comb_total_pass_list, columns=['sag_id', 'subcontig_id', 'contig_id'])

    tetra_df_dict = {'gmm':gmm_df, 'svm':svm_df, 'iso':iso_df, 'comb':comb_df}
    #tetra_df_dict = {'gmm':gmm_df, 'svm':svm_df, 'comb':comb_df}

    for tetra_id in tetra_df_dict:
        tetra_df = tetra_df_dict[tetra_id]
        #mg_id, mg_headers, mg_subs = mg_subcontigs

        # Count # of subcontigs recruited to each SAG
        gmm_cnt_df = tetra_df.groupby(['sag_id', 'contig_id']).count().reset_index()
        gmm_cnt_df.columns = ['sag_id', 'contig_id', 'subcontig_recruits']
        # Build subcontig count for each MG contig
        mg_contig_list = [x.rsplit('_', 1)[0] for x in mg_headers]
        mg_tot_df = pd.DataFrame(zip(mg_contig_list, mg_headers),
                                 columns=['contig_id', 'subcontig_id'])
        mg_tot_cnt_df = mg_tot_df.groupby(['contig_id']).count().reset_index()
        mg_tot_cnt_df.columns = ['contig_id', 'subcontig_total']
        mg_recruit_df = gmm_cnt_df.merge(mg_tot_cnt_df, how='left', on='contig_id')
        mg_recruit_df['percent_recruited'] = mg_recruit_df['subcontig_recruits'] / \
                                             mg_recruit_df['subcontig_total']
        mg_recruit_df.sort_values(by='percent_recruited', ascending=False, inplace=True)
        # Only pass contigs that have the magjority of subcontigs recruited (>= N%)
        mg_recruit_filter_df = mg_recruit_df.loc[mg_recruit_df['percent_recruited'] >= float(gmm_per_pass)]
        mg_contig_per_max_df = mg_recruit_filter_df.groupby(['contig_id'])[
            'percent_recruited'].max().reset_index()
        mg_contig_per_max_df.columns = ['contig_id', 'percent_max']
        mg_recruit_max_df = mg_recruit_filter_df.merge(mg_contig_per_max_df, how='left',
                                                       on='contig_id')
        # Now pass contigs that have the maximum recruit % of subcontigs
        mg_max_only_df = mg_recruit_max_df.loc[mg_recruit_max_df['percent_recruited'] >=
                                               mg_recruit_max_df['percent_max']
                                               ]
        mg_max_only_df.to_csv(o_join(tra_path, mg_id + '.' + tetra_id + '.tra_trimmed_recruits.tsv'), sep='\t', index=False)

        tetra_df_dict[tetra_id] = mg_max_only_df


    return tetra_df_dict


'''
    ##################################
    # build scatterplot to viz the GMM
    sag_xy_df = sag_umap_df.iloc[:,0:2].copy()
    mg_xy_df = mg_umap_df.iloc[:,0:2].copy()
    sag_xy_df['isSAG'] = 'SAG'
    mg_xy_df['isSAG'] = ['Tetra-Recruit' if x in list(gmm_pass_df.index.values)
                            else 'MG' for x in mg_xy_df.index.values
                            ]
    recruits_xy_df = mg_xy_df[mg_xy_df['isSAG'] == 'Tetra-Recruit']
    mg_xy_df = mg_xy_df[mg_xy_df['isSAG'] == 'MG']
    xy_df = pd.concat([mg_xy_df, sag_xy_df, recruits_xy_df])
    xy_df.to_csv(join(tra_path, sag_id + '.GMM_plot.tsv'), sep='\t')
    sv_plot = join(tra_path, sag_id + '.GMM_plot.png')
    ax = sns.scatterplot(x='pc1', y='pc2', data=xy_df, hue='isSAG',
                            alpha=0.4, edgecolor='none')
    plt.gca().set_aspect('equal', 'datalim')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(sv_plot, bbox_inches="tight")
    plt.clf()
    ##################################
'''
