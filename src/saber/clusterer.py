#!/usr/bin/env python

import logging
import multiprocessing
import warnings
from os.path import join as o_join
from pathlib import Path

import hdbscan
import pandas as pd
import umap
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

import saber.utilities as s_utils

warnings.simplefilter("error", category=UserWarning)
warnings.filterwarnings('ignore', '.*sparsity.*', )
warnings.simplefilter("ignore", category=DeprecationWarning)


def runOCSVM(tc_df, mg_df, tc_id, n, gam):
    # fit OCSVM
    clf = svm.OneClassSVM(nu=n, gamma=gam)
    clf.fit(tc_df.values)
    mg_pred = clf.predict(mg_df.values)
    contig_id_list = [x.rsplit('_', 1)[0] for x in mg_df.index.values]
    pred_df = pd.DataFrame(zip(mg_df.index.values, contig_id_list, mg_pred),
                           columns=['subcontig_id', 'contig_id', 'pred']
                           )
    pred_df['nu'] = n
    pred_df['gamma'] = gam
    pred_df['sag_id'] = tc_id
    pred_df = pred_df[['sag_id', 'nu', 'gamma', 'subcontig_id', 'contig_id', 'pred']]

    return pred_df


def runKMEANS(recruit_contigs_df, sag_id, std_merge_df):
    temp_cat_df = std_merge_df.copy()
    last_len = 0
    while temp_cat_df.shape[0] != last_len:
        last_len = temp_cat_df.shape[0]
        clusters = 10 if last_len >= 10 else last_len
        kmeans = MiniBatchKMeans(n_clusters=clusters, random_state=42).fit(temp_cat_df.values)
        clust_labels = kmeans.labels_
        clust_df = pd.DataFrame(zip(temp_cat_df.index.values, clust_labels),
                                columns=['subcontig_id', 'kmeans_clust']
                                )
        recruit_clust_df = clust_df.loc[clust_df['subcontig_id'].isin(list(recruit_contigs_df.index))]
        subset_clust_df = clust_df.loc[clust_df['kmeans_clust'].isin(
            list(recruit_clust_df['kmeans_clust'].unique())
        )]
        subset_clust_df['kmeans_pred'] = 1
        temp_cat_df = temp_cat_df.loc[temp_cat_df.index.isin(list(subset_clust_df['subcontig_id']))]
    cat_clust_df = subset_clust_df.copy()
    std_id_df = pd.DataFrame(std_merge_df.index.values, columns=['subcontig_id'])
    std_id_df['contig_id'] = [x.rsplit('_', 1)[0] for x in std_id_df['subcontig_id']]
    cat_clust_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cat_clust_df['subcontig_id']]
    sub_std_df = std_id_df.loc[std_id_df['contig_id'].isin(list(cat_clust_df['contig_id']))]
    std_clust_df = sub_std_df.merge(cat_clust_df, on=['subcontig_id', 'contig_id'], how='outer')
    std_clust_df.fillna(-1, inplace=True)
    pred_df = std_clust_df[['subcontig_id', 'contig_id', 'kmeans_pred']]
    val_perc = pred_df.groupby('contig_id')['kmeans_pred'
    ].value_counts(normalize=True).reset_index(name='percent')
    pos_perc = val_perc.loc[val_perc['kmeans_pred'] == 1]
    major_df = pos_perc.loc[pos_perc['percent'] >= 0.51]
    major_pred_df = pred_df.loc[pred_df['contig_id'].isin(major_df['contig_id'])]
    std_clust_pred_df = std_clust_df.merge(major_pred_df, on=['subcontig_id', 'contig_id'],
                                           how='left'
                                           )
    filter_clust_pred_df = std_clust_pred_df.loc[std_clust_pred_df['kmeans_pred_x'] == 1]
    kmeans_pass_list = []
    for md_nm in major_pred_df['subcontig_id']:
        kmeans_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
    return kmeans_pass_list, filter_clust_pred_df


def recruitOCSVM(p):
    m_df, sub_trusted_df, sag_id, nu, gamma = p
    tc_contig_list = list(sub_trusted_df['contig_id'].unique())
    m_df['contig_id'] = [x.rsplit('_', 1)[0] for x in m_df.index.values]
    tc_feat_df = m_df.query('contig_id in @tc_contig_list')
    mg_feat_df = m_df.copy()  # .query('contig_id in @sub_contig_list')
    mg_feat_df.drop(columns=['contig_id'], inplace=True)
    tc_feat_df.drop(columns=['contig_id'], inplace=True)
    major_df = False
    if (tc_feat_df.shape[0] != 0) & (mg_feat_df.shape[0] != 0):
        '''
        # removed from preprocessing
        # Run KMEANS first
        kmeans_pass_list, kclusters_df = runKMEANS(tc_feat_df, sag_id, mg_feat_df)
        kmeans_pass_df = pd.DataFrame(kmeans_pass_list,
                                      columns=['sag_id', 'subcontig_id', 'contig_id']
                                      )
        nonrecruit_kmeans_df = mg_feat_df.loc[kmeans_pass_df['subcontig_id']]
        '''
        ocsvm_recruit_df = runOCSVM(tc_feat_df, mg_feat_df, sag_id, nu, gamma)
        val_perc = ocsvm_recruit_df.groupby('contig_id')['pred'].value_counts(
            normalize=True).reset_index(name='percent')
        pos_perc = val_perc.query('pred == 1')
        major_df = pos_perc.copy()
        major_df['sag_id'] = sag_id

    return sag_id, major_df

def runClusterer(mg_id, tmp_path, clst_path, cov_file, tetra_file, minhash_dict,
                 denovo_min_clust, denovo_min_samp, anchor_min_clust, anchor_min_samp,
                 nu, gamma, jaccard, nthreads
                 ):  # TODO: need to add multithreading where ever possible
    # Get the MinHash recruits
    if minhash_dict:
        mh_trusted_df = minhash_dict[201]
        mh_trusted_df.rename(columns={'q_contig_id': 'contig_id'}, inplace=True)
        mh_best_df = mh_trusted_df.query(f'jacc_sim == {jaccard}')
    # Convert CovM to UMAP feature table
    set_init = 'spectral'
    merged_emb = Path(o_join(tmp_path, mg_id + '.merged_emb.tsv'))
    if not merged_emb.is_file():
        cov_emb = Path(o_join(tmp_path, mg_id + '.covm_emb.tsv'))
        print('Building embedding for Coverage...')
        cov_df = pd.read_csv(cov_file, header=0, sep='\t', index_col='subcontig_id')
        cov_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cov_df.index]
        #mh_contig_list = list(mh_trusted_df['contig_id'].unique())
        mh_cov_df = cov_df.copy() #.query('contig_id == @mh_contig_list')
        del mh_cov_df['contig_id']
        n_neighbors = 20
        # COV sometimes crashes when init='spectral', trying higher NN value for 2-stage DR
        try:
            clusterable_embedding = umap.UMAP(random_state=42).fit_transform(mh_cov_df)
                                              #n_neighbors=n_neighbors, min_dist=0.1,
                                              #n_components=len(mh_cov_df.columns),
                                              #random_state=42, metric='manhattan', init=set_init
                                              #).fit_transform(mh_cov_df)
        except:
            try:
                print('Spectral Initialization Failed!')
                print('Running 2-stage DR to provide Initial Embedding...')
                tmp_nn = 50
                tmp_embedding = umap.UMAP(random_state=42).fit_transform(mh_cov_df)
                                          #n_neighbors=tmp_nn, min_dist=0.1,
                                          #n_components=len(mh_cov_df.columns),
                                          #random_state=42, metric='manhattan',
                                          #init=set_init
                                          #).fit_transform(mh_cov_df)
                print('Initialization worked with n_neighbors=50, moving to Stage 2...')
                clusterable_embedding = umap.UMAP(random_state=42).fit_transform(mh_cov_df)
                                                  #n_neighbors=n_neighbors, min_dist=0.1,
                                                  #n_components=len(mh_cov_df.columns),
                                                  #random_state=42, metric='manhattan',
                                                  #init=tmp_embedding
                                                  #).fit_transform(mh_cov_df)
            except:
                print('2-Stage Initialization Failed!')
                print('Running PCA to provide Initial Embedding...')
                pca = PCA(n_components=len(mh_cov_df.columns))
                pca_emb = pca.fit_transform(mh_cov_df)
                print('Fitting Coverage Data with Anchors...')
                clusterable_embedding = umap.UMAP(random_state=42).fit_transform(mh_cov_df)
                                                  #n_neighbors=n_neighbors, min_dist=0.1,
                                                  #n_components=len(mh_cov_df.columns),
                                                  #random_state=42, metric='manhattan',
                                                  #init=pca_emb
                                                  #).fit_transform(mh_cov_df)
        umap_feat_df = pd.DataFrame(clusterable_embedding, index=mh_cov_df.index.values)
        umap_feat_df.reset_index(inplace=True)
        umap_feat_df.rename(columns={'index': 'subcontig_id'}, inplace=True)
        umap_feat_df.to_csv(cov_emb, sep='\t', index=False)

        # Convert Tetra to UMAP feature table
        tetra_emb = Path(o_join(tmp_path, mg_id + '.tetra_emb.tsv'))
        print('Building embedding for Tetra Hz...')
        tetra_df = pd.read_csv(tetra_file, header=0, sep='\t', index_col='contig_id')
        tetra_df['contig_id'] = [x.rsplit('_', 1)[0] for x in tetra_df.index]
        #mh_contig_list = list(mh_trusted_df['contig_id'].unique())
        mh_tetra_df = tetra_df.copy() #.query('contig_id == @mh_contig_list')
        del mh_tetra_df['contig_id']
        #n_neighbors = 10
        try:
            clusterable_embedding = umap.UMAP(random_state=42).fit_transform(mh_tetra_df)
                                              #n_neighbors=n_neighbors, min_dist=0.1, n_components=3,
                                              #random_state=42, metric='manhattan', init=set_init
                                              #).fit_transform(mh_tetra_df)
        except:
            try:
                print('Spectral Initialization Failed!')
                print('Running 2-stage DR to provide Initial Embedding...')
                tmp_nn = 50
                tmp_embedding = umap.UMAP(random_state=42).fit_transform(mh_tetra_df)
                                          #n_neighbors=tmp_nn, min_dist=0.1,
                                          #n_components=3,
                                          ##random_state=42, metric='manhattan',
                                          #init=set_init
                                          #).fit_transform(mh_tetra_df)
                print('Initialization worked with n_neighbors=50, moving to Stage 2...')
                clusterable_embedding = umap.UMAP(random_state=42).fit_transform(mh_tetra_df)
                                                  #n_neighbors=n_neighbors, min_dist=0.1,
                                                  #n_components=3,
                                                  #random_state=42, metric='manhattan',
                                                  #init=tmp_embedding
                                                  #).fit_transform(mh_tetra_df)
            except:
                print('2-Stage Initialization Failed!')
                print('Running PCA to provide Initial Embedding...')
                pca = PCA(n_components=3)
                pca_emb = pca.fit_transform(mh_tetra_df)
                print('Fitting Coverage Data with Anchors...')
                clusterable_embedding = umap.UMAP(random_state=42).fit_transform(mh_tetra_df)
                                                  #n_neighbors=n_neighbors, min_dist=0.1,
                                                  #n_components=3,
                                                  #random_state=42, metric='manhattan',
                                                  #init=pca_emb
                                                  #).fit_transform(mh_tetra_df)
        umap_feat_df = pd.DataFrame(clusterable_embedding, index=mh_tetra_df.index.values)
        umap_feat_df.reset_index(inplace=True)
        umap_feat_df.rename(columns={'index': 'subcontig_id'}, inplace=True)
        umap_feat_df.to_csv(tetra_emb, sep='\t', index=False)

        # Merge Coverage and Tetra Embeddings
        print('Merging Tetra and Coverage Embeddings...')
        tetra_feat_df = pd.read_csv(tetra_emb, sep='\t', header=0, index_col='subcontig_id')
        tetra_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in tetra_feat_df.index.values]
        tetra_feat_df.columns = [str(x) + '_tetra' for x in tetra_feat_df.columns]
        # load covm file
        cov_feat_df = pd.read_csv(cov_emb, sep='\t', header=0)
        cov_feat_df.rename(columns={'contigName': 'subcontig_id'}, inplace=True)
        cov_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cov_feat_df['subcontig_id']]
        cov_feat_df.set_index('subcontig_id', inplace=True)
        cov_feat_df.columns = [str(x) + '_cov' for x in cov_feat_df.columns]
        merge_df = cov_feat_df.merge(tetra_feat_df, left_index=True, right_index=True, how='left')
        merge_df.drop(columns=['contig_id_tetra', 'contig_id_cov'], inplace=True)
        tetra_feat_df.drop(columns=['contig_id_tetra'], inplace=True)
        cov_feat_df.drop(columns=['contig_id_cov'], inplace=True)
        merge_df.to_csv(merged_emb, sep='\t')

    denovo_out_file = Path(o_join(clst_path, mg_id + '.denovo_hdbscan.tsv'))
    if not denovo_out_file.is_file():
        print('Performing De Novo Clustering...')
        merge_df = pd.read_csv(merged_emb, sep='\t', header=0, index_col='subcontig_id')
        clusterer = hdbscan.HDBSCAN(min_cluster_size=denovo_min_clust, prediction_data=True,
                                    min_samples=denovo_min_samp, core_dist_n_jobs=nthreads
                                    ).fit(merge_df.values)

        cluster_labels = clusterer.labels_
        cluster_probs = clusterer.probabilities_
        cluster_outlier = clusterer.outlier_scores_

        cluster_df = pd.DataFrame(zip(merge_df.index.values, cluster_labels, cluster_probs,
                                      cluster_outlier),
                                  columns=['subcontig_id', 'label', 'probabilities',
                                           'outlier_score']
                                  )
        cluster_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cluster_df['subcontig_id']]
        cluster_df.to_csv(denovo_out_file, sep='\t', index=False)
    else:
        print('Loading De Novo Clusters...')
        cluster_df = pd.read_csv(denovo_out_file, header=0, sep='\t')
    denovo_out_file = Path(o_join(clst_path, mg_id + '.denovo_clusters.tsv'))
    noise_out_file = Path(o_join(clst_path, mg_id + '.denovo_noise.tsv'))
    if not denovo_out_file.is_file():
        print('Denoising Clusters...')
        pool = multiprocessing.Pool(processes=nthreads)
        arg_list = []
        for contig in tqdm(list(cluster_df['contig_id'].unique())):
            sub_df = cluster_df.query('contig_id == @contig')
            arg_list.append([sub_df, contig])
        ns_ratio_list = []
        results = pool.imap_unordered(denoise_clust, arg_list)
        for i, output in tqdm(enumerate(results, 1)):
            ns_ratio_list.append(output)
        pool.close()
        pool.join()
        ns_ratio_df = pd.DataFrame(ns_ratio_list, columns=['contig_id', 'best_label'])
        cluster_ns_df = cluster_df.merge(ns_ratio_df, on='contig_id', how='left')
        denovo_clusters_df = cluster_ns_df.query('best_label != -1')  # 'best_prob >= 0.51')
        noise_df = cluster_ns_df.query('best_label == -1')  # 'best_prob < 0.51')
        if denovo_clusters_df.empty:
            #  TODO: fix this or print a warning message to user :)
            denovo_clusters_df = noise_df.copy()
        denovo_clusters_df.to_csv(denovo_out_file, sep='\t', index=False)
        noise_df.to_csv(noise_out_file, sep='\t', index=False)
    else:
        print('Loading Cleaned De Novo Clusters...')
        denovo_clusters_df = pd.read_csv(denovo_out_file, header=0, sep='\t')
        noise_df = pd.read_csv(noise_out_file, header=0, sep='\t')

    ######################################
    ########## ANCHORED BINNING ##########
    ######################################
    if minhash_dict:
        # Run HDSCAN ANCHORED
        trust_anchors_file = Path(o_join(clst_path, mg_id + '.hdbscan_anchors.tsv'))
        if not trust_anchors_file.is_file():
            print('Anchored Binning Starting with Trusted Contigs...')
            print('Clustering with HDBSCAN and Anchored Settings...')
            merge_df = pd.read_csv(merged_emb, sep='\t', header=0, index_col='subcontig_id')
            clusterer = hdbscan.HDBSCAN(min_cluster_size=anchor_min_clust, prediction_data=True,
                                        min_samples=anchor_min_samp, core_dist_n_jobs=nthreads
                                        ).fit(merge_df.values)

            cluster_labels = clusterer.labels_
            cluster_probs = clusterer.probabilities_
            cluster_outlier = clusterer.outlier_scores_

            cluster_df = pd.DataFrame(zip(merge_df.index.values, cluster_labels, cluster_probs,
                                          cluster_outlier),
                                      columns=['subcontig_id', 'label', 'probabilities',
                                               'outlier_score']
                                      )
            cluster_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cluster_df['subcontig_id']]
            cluster_df.to_csv(trust_anchors_file, sep='\t', index=False)
        else:
            print('Loading HDBSCAN Anchored Clusters...')
            cluster_df = pd.read_csv(trust_anchors_file, header=0, sep='\t')
        hdbscan_out_file = Path(o_join(clst_path, mg_id + '.hdbscan_clusters.tsv'))
        noise_out_file = Path(o_join(clst_path, mg_id + '.hdbscan_noise.tsv'))
        if not hdbscan_out_file.is_file():
            print('Denoising Clusters...')
            pool = multiprocessing.Pool(processes=nthreads)
            arg_list = []
            for contig in tqdm(list(cluster_df['contig_id'].unique())):
                sub_df = cluster_df.query('contig_id == @contig')
                arg_list.append([sub_df, contig])
            ns_ratio_list = []
            results = pool.imap_unordered(denoise_clust, arg_list)
            for i, output in tqdm(enumerate(results, 1)):
                ns_ratio_list.append(output)
            pool.close()
            pool.join()
            ns_ratio_df = pd.DataFrame(ns_ratio_list, columns=['contig_id', 'best_label'])
            cluster_ns_df = cluster_df.merge(ns_ratio_df, on='contig_id', how='left')
            no_noise_df = cluster_ns_df.query('best_label != -1')  # 'best_prob >= 0.51')
            noise_df = cluster_ns_df.query('best_label == -1')  # 'best_prob < 0.51')
            #if no_noise_df.empty:
            #    #  TODO: fix this or print a warning message to user :)
            #    no_noise_df = noise_df.copy()
            
            # Group clustered and noise contigs by trusted contigs
            print('Re-grouping with Trusted Contigs...')
            pool = multiprocessing.Pool(processes=nthreads)
            arg_list = []
            for sag_id in tqdm(mh_best_df['sag_id'].unique()):
                sub_mh_df = mh_best_df.query('sag_id == @sag_id')
                arg_list.append([sub_mh_df, no_noise_df, noise_df, sag_id])
            label_max_list = []
            contig_max_list = []
            results = pool.imap_unordered(trust_clust, arg_list)
            for i, output in tqdm(enumerate(results, 1)):
                if isinstance(output[0], pd.DataFrame):
                    label_max_list.append(output[0])
                if isinstance(output[1], pd.DataFrame):
                    contig_max_list.append(output[1])
            pool.close()
            pool.join()
            if label_max_list:
                sag_label_df = pd.concat(label_max_list)
            else:
                sag_label_df = pd.DataFrame(columns=['sag_id', 'contig_id', 'anch_cnt'])
            if contig_max_list:
                sag_contig_df = pd.concat(contig_max_list)
            else:
                sag_contig_df = pd.DataFrame(columns=['sag_id', 'contig_id'])
            sag_label_best_df = sag_label_df.sort_values(by='anch_cnt', ascending=False
                                                         ).drop_duplicates(subset='best_label')
            print('Extracting Best Clusters...')
            hdbscan_label_dict = {}
            for index, row in tqdm(sag_label_best_df.iterrows()):
                sag_id = row['sag_id']
                best_label = row['best_label']
                sub_sag_label_df = sag_label_df.query('sag_id == @sag_id and '
                                                      'best_label == @best_label'
                                                      )
                hdbscan_label_dict[sag_id] = sub_sag_label_df

            print('Building HDBSCAN Anchored Clusters...')
            trust_recruit_list = []
            count = 0
            for sag_id in tqdm(mh_best_df['sag_id'].unique()):
                trust_cols = ['sag_id', 'contig_id']
                sub_trusted_df = mh_best_df.query('sag_id == @sag_id')[trust_cols]
                subs_list = [sub_trusted_df]
                #subs_list = []
                if sag_id in hdbscan_label_dict.keys():
                    sub_label_df = hdbscan_label_dict[sag_id][trust_cols]
                    subs_list.append(sub_label_df)
                if sag_id in sag_contig_df['sag_id']:
                    sub_contig_df = sag_contig_df.query('sag_id == @sag_id')[trust_cols]
                    subs_list.append(sub_contig_df)
                if len(subs_list) > 1:
                    hdbscan_cat_df = pd.concat(subs_list).drop_duplicates()
                else:
                    hdbscan_cat_df = sub_trusted_df.drop_duplicates()
                trust_recruit_list.append(hdbscan_cat_df)
            trust_recruit_df = pd.concat(trust_recruit_list)
            trust_recruit_df.rename(columns={'sag_id': 'best_label'}, inplace=True)
            trust_recruit_df.to_csv(hdbscan_out_file, sep='\t', index=False)
            noise_df.to_csv(noise_out_file, sep='\t', index=False)
        elif hdbscan_out_file.is_file():
            print('HDBSCAN Anchored Clusters already exist...')
            trust_recruit_df = pd.read_csv(hdbscan_out_file, sep='\t', header=0)
    else:
        print('No Trusted Contigs Provided...')
        trust_recruit_df = False

    if minhash_dict:
        # Run OC-SVM recruiting
        ocsvm_out_file = Path(o_join(clst_path, mg_id + '.ocsvm_clusters.tsv'))
        if not ocsvm_out_file.is_file():
            print('Performing Anchored Recruitment with OC-SVM...')
            merge_df = pd.read_csv(merged_emb, sep='\t', header=0, index_col='subcontig_id')
            print('Running OC-SVM algorithm...')
            pool = multiprocessing.Pool(processes=nthreads)
            arg_list = []
            oc_sag_list = list(mh_best_df['sag_id'].unique())
            for sag_id in tqdm(oc_sag_list):
                sub_mh_df = mh_best_df.query('sag_id == @sag_id')
                arg_list.append([merge_df, sub_mh_df, sag_id, nu, gamma])
            ocsvm_recruit_list = []
            ocsvm_recruit_dict = {}
            results = pool.imap_unordered(recruitOCSVM, arg_list)
            for i, output in tqdm(enumerate(results, 1)):
                sag_id, ocsvm_recruits = output
                if isinstance(ocsvm_recruits, pd.DataFrame):
                    ocsvm_recruit_list.append(ocsvm_recruits)
                    ocsvm_recruit_dict[sag_id] = ocsvm_recruits
            pool.close()
            pool.join()
            ocsvm_contig_df = pd.concat(ocsvm_recruit_list)
            ocsvm_contig_best_df = ocsvm_contig_df.sort_values(by='percent', ascending=False
                                                               ).drop_duplicates(subset='contig_id')
            print('Compiling OC-SVM recruited contigs...')
            ocsvm_contig_dict = {}
            for index, row in tqdm(list(ocsvm_contig_best_df.iterrows())):
                sag_id = row['sag_id']
                percent = row['percent']
                sub_ocsvm_df = ocsvm_recruit_dict[sag_id]
                sub_sag_contig_df = sub_ocsvm_df.query('percent >= @percent')
                ocsvm_contig_dict[sag_id] = sub_sag_contig_df

            print('Building OC-SVM Clusters...')
            ocsvm_clust_list = []
            for sag_id in tqdm(oc_sag_list):
                trust_cols = ['sag_id', 'contig_id']
                sub_trusted_df = mh_best_df.query('sag_id == @sag_id')[trust_cols]
                if sag_id in ocsvm_contig_dict.keys():
                    sub_ocsvm_df = ocsvm_contig_dict[sag_id][trust_cols]
                    ocsvm_cat_df = pd.concat([sub_trusted_df, sub_ocsvm_df]).drop_duplicates()
                    ocsvm_clust_list.append(ocsvm_cat_df)
                else:
                    ocsvm_cat_df = sub_trusted_df.drop_duplicates()
                    ocsvm_clust_list.append(ocsvm_cat_df)
            ocsvm_clust_df = pd.concat(ocsvm_clust_list)
            ocsvm_clust_df.rename(columns={'sag_id': 'best_label'}, inplace=True)
            ocsvm_clust_df.to_csv(ocsvm_out_file, sep='\t', index=False)
        elif ocsvm_out_file.is_file():
            print('OC-SVM Clusters already exist...')
            ocsvm_clust_df = pd.read_csv(ocsvm_out_file, sep='\t', header=0)
    else:
        ocsvm_clust_df = False

    if minhash_dict:
        # Find intersection of HDBSCAN and OC-SVM
        inter_out_file = Path(o_join(clst_path, mg_id + '.inter_clusters.tsv'))
        if not inter_out_file.is_file():
            print('Combining Recruits from HDBSCAN and OC-SVM...')
            inter_clust_list = []
            #trust_labels = list(trust_recruit_df['best_label'].unique())
            trust_labels = list(mh_best_df['sag_id'].unique())
            for best_label in tqdm(trust_labels):
                sub_mh_best_df = mh_best_df.query('sag_id == @best_label')[['sag_id', 'contig_id']]
                sub_mh_best_df.columns = ['best_label', 'contig_id']
                sub_minhash_df = mh_trusted_df.query('sag_id == @best_label')[['sag_id', 'contig_id']]
                sub_scan_df = trust_recruit_df.query('best_label == @best_label')
                sub_svm_df = ocsvm_clust_df.query('best_label == @best_label')
                mh_hdb_inter = set(sub_minhash_df['contig_id']).intersection(set(sub_scan_df['contig_id']))
                mh_svm_inter = set(sub_minhash_df['contig_id']).intersection(set(sub_svm_df['contig_id']))
                hdb_svm_inter = set(sub_scan_df['contig_id']).intersection(set(sub_svm_df['contig_id']))
                inter_list = [mh_hdb_inter, mh_svm_inter, hdb_svm_inter]
                intersect = list(set().union(*inter_list))
                best_inter_list = [(best_label, x) for x in intersect]
                best_inter_df = pd.DataFrame(best_inter_list, columns=['best_label', 'contig_id'])
                best_concat_df = pd.concat([best_inter_df, sub_mh_best_df]).drop_duplicates()
                inter_clust_list.append(best_concat_df)
            inter_clust_df = pd.concat(inter_clust_list)
            inter_clust_df.to_csv(inter_out_file, sep='\t', index=False)
        elif inter_out_file.is_file():
            print('Combined Clusters already exist...')
            inter_clust_df = pd.read_csv(inter_out_file, sep='\t', header=0)
    else:
        inter_clust_df = False

    logging.info('Cleaning up intermediate files...\n')
    for s in ["*.denovo.covm_emb.tsv", "*.denovo.tetra_emb.tsv",
              "*.anchored.covm_emb.tsv", "*.anchored.tetra_emb.tsv",
              "*.subcontigs.*"
              ]:
        s_utils.runCleaner(clst_path, s)

    return denovo_clusters_df, trust_recruit_df, ocsvm_clust_df, inter_clust_df


def sag_compare(p):
    mh_trusted_df, s1, s2 = p
    s1_sub = s1.rsplit('.', 1)[0]
    s2_sub = s2.rsplit('.', 1)[0]
    s1_contigs = set(mh_trusted_df.query('sag_id == @s1')['contig_id'])
    s2_contigs = set(mh_trusted_df.query('sag_id == @s2')['contig_id'])
    s_inter = len(s1_contigs.intersection(s2_contigs))
    s_union = len(s1_contigs.union(s2_contigs))
    s_jacc = s_inter / s_union
    if (s1_sub == s2_sub) & (s_inter > 3):
        sag_comp_tup = (True, s_inter, s_union, s_jacc)
    elif (s1_sub != s2_sub) & (s_inter > 3):
        sag_comp_tup = (False, s_inter, s_union, s_jacc)
    else:
        sag_comp_tup = (None, s_inter, s_union, s_jacc)

    return s1, s2, sag_comp_tup


def trust_build(p):
    mh_trusted_df, sag_denovo_df, sag_noise_df, sag_id = p
    trust_cols = ['sag_id', 'contig_id']
    sub_trusted_df = mh_trusted_df.query('sag_id == @sag_id')[trust_cols]
    sub_denovo_df = sag_denovo_df.query('sag_id == @sag_id')[trust_cols]
    sub_noise_df = sag_noise_df.query('sag_id == @sag_id')[trust_cols]
    trust_cat_df = pd.concat([sub_trusted_df, sub_denovo_df, sub_noise_df]
                             ).drop_duplicates()
    return trust_cat_df


def trust_clust(p):
    sub_trusted_df, no_noise_df, noise_df, sag_id = p
    # Subset trusted contigs by SAG
    trusted_contigs_list = sub_trusted_df['contig_id'].unique()
    # Gather trusted contigs subset from cluster and noise DFs
    sub_nonise_df = no_noise_df.query('contig_id in @trusted_contigs_list')
    if sub_nonise_df.shape[0] != 0:
        # Gather all contigs associated with trusted clusters
        trust_clust_list = list(sub_nonise_df['best_label'].unique())
        sub_label_df = no_noise_df.query('best_label in @trust_clust_list')
        lab2anch = {}
        for label in sub_label_df['best_label'].unique():
            label_nonise_df = sub_nonise_df.query('best_label == @label')
            anch_cnt = label_nonise_df.shape[0]
            if anch_cnt != '':
                lab2anch[label] = anch_cnt
        sub_label_df['sag_id'] = sag_id
        sub_label_df['anch_cnt'] = [lab2anch[x] for x in sub_label_df['best_label']]
    else:
        sub_label_df = None
    # Get max jaccard for noise labeled contigs
    sub_noise_df = noise_df.query('contig_id in @trusted_contigs_list')
    if sub_noise_df.shape[0] != 0:
        sub_noise_df['sag_id'] = sag_id
        sub_noise_df['anch_cnt'] = 1
    else:
        sub_noise_df = None

    return sub_label_df, sub_noise_df


def denoise_clust(p):
    sub_df, contig = p
    noise_cnt = sub_df.query('label == -1').shape[0]
    signal_cnt = sub_df.query('label != -1').shape[0]
    ns_ratio = (noise_cnt / (noise_cnt + signal_cnt)) * 100
    #if ns_ratio < 51:
    if signal_cnt != 0:
        prob_df = sub_df.groupby(['label'])[['probabilities']].max().reset_index()
        best_ind = prob_df['probabilities'].argmax()
        best_label = prob_df['label'].iloc[best_ind]
        best_prob = prob_df['probabilities'].iloc[best_ind]
    else:
        best_label = -1
    return ([contig, best_label])


def match_contigs(p):
    s1, s2, s1_col, s2_col = p
    s1_set = set([i for i, e in enumerate(s1_col) if e == 1])
    s2_set = set([i for i, e in enumerate(s2_col) if e == 1])
    s1_s2 = len(s1_set.intersection(s2_set))
    if s1_s2 > 3:
        match_count = 1
    else:
        match_count = 0
    return s1, s2, match_count
