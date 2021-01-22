# import matplotlib
# matplotlib.use('Agg')
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import Pipeline
# from umap import UMAP
import sys

import pandas as pd
from sklearn import svm

# from functools import reduce
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
# import warnings
# import sklearn.exceptions
# warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
# from song.song import SONG
# from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc  # , f1_score, matthews_corrcoef
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import multiprocessing
from sklearn.cluster import MiniBatchKMeans
import logging


def recruit_stats(p):
    sag_id, gam, n, subcontig_id_list, contig_id_list, exact_truth, strain_truth, pred = p

    pred_df = pd.DataFrame(zip(subcontig_id_list, contig_id_list, pred),
                           columns=['subcontig_id', 'contig_id', 'pred']
                           )
    pred_df['sag_id'] = sag_id
    pred_df['gamma'] = gam
    pred_df['nu'] = n

    pred_df = pred_df[['sag_id', 'nu', 'gamma', 'subcontig_id', 'contig_id', 'pred']]

    val_perc = pred_df.groupby('contig_id')['pred'].value_counts(
        normalize=True).reset_index(name='precent')
    pos_perc = val_perc.loc[val_perc['pred'] == 1]
    major_df = pos_perc.loc[pos_perc['precent'] >= 0.51]
    major_pred = [1 if x in list(major_df['contig_id']) else -1
                  for x in pred_df['contig_id']
                  ]
    pos_pred_list = list(set(pred_df.loc[pred_df['pred'] == 1]['contig_id']))
    all_pred = [1 if x in pos_pred_list else -1
                for x in pred_df['contig_id']
                ]
    pred_df['all_pred'] = all_pred
    pred_df['major_pred'] = major_pred
    pred_df['truth'] = exact_truth
    pred_df['truth_strain'] = strain_truth
    # ALL Recruits
    # calculate for hybrid exact/strain-level matches
    TP = calc_tp(pred_df['truth'], pred_df['all_pred'])
    FP = calc_fp(pred_df['truth_strain'], pred_df['all_pred'])
    TN = calc_tn(pred_df['truth'], pred_df['all_pred'])
    FN = calc_fn(pred_df['truth'], pred_df['all_pred'])
    all_str_list = calc_stats(sag_id, 'strain', 'all', gam, n, TP, FP, TN, FN,
                              pred_df['truth_strain'], pred_df['all_pred']
                              )
    # ALL Recruits
    # calculate for exact-level match
    TP = calc_tp(pred_df['truth'], pred_df['all_pred'])
    FP = calc_fp(pred_df['truth'], pred_df['all_pred'])
    TN = calc_tn(pred_df['truth'], pred_df['all_pred'])
    FN = calc_fn(pred_df['truth'], pred_df['all_pred'])
    all_x_list = calc_stats(sag_id, 'exact', 'all', gam, n, TP, FP, TN, FN,
                            pred_df['truth'], pred_df['all_pred']
                            )

    # Majority-Rule Recruits
    # calculate for hybrid exact/strain-level matches
    TP = calc_tp(pred_df['truth'], pred_df['major_pred'])
    FP = calc_fp(pred_df['truth_strain'], pred_df['major_pred'])
    TN = calc_tn(pred_df['truth'], pred_df['major_pred'])
    FN = calc_fn(pred_df['truth'], pred_df['major_pred'])
    maj_str_list = calc_stats(sag_id, 'strain', 'majority', gam, n, TP, FP, TN, FN,
                              pred_df['truth_strain'], pred_df['major_pred']
                              )
    # Majority-Rule Recruits
    # calculate for exact-level match
    TP = calc_tp(pred_df['truth'], pred_df['major_pred'])
    FP = calc_fp(pred_df['truth'], pred_df['major_pred'])
    TN = calc_tn(pred_df['truth'], pred_df['major_pred'])
    FN = calc_fn(pred_df['truth'], pred_df['major_pred'])
    maj_x_list = calc_stats(sag_id, 'exact', 'majority', gam, n, TP, FP, TN, FN,
                            pred_df['truth'], pred_df['major_pred']
                            )
    filter_pred_df = pred_df.loc[pred_df['major_pred'] == 1]

    return all_str_list, all_x_list, maj_str_list, maj_x_list, filter_pred_df


def calc_tp(y_truth, y_pred):
    tp_list = pd.Series([1 if ((x[0] == 1) & (x[1] == 1)) else 0 for x in zip(y_truth, y_pred)])
    TP = tp_list.sum()

    return TP


def calc_fp(y_truth, y_pred):
    fp_list = pd.Series([1 if ((x[0] == -1) & (x[1] == 1)) else 0 for x in zip(y_truth, y_pred)])
    FP = fp_list.sum()

    return FP


def calc_tn(y_truth, y_pred):
    tn_list = pd.Series([1 if ((x[0] == -1) & (x[1] == -1)) else 0 for x in zip(y_truth, y_pred)])
    TN = tn_list.sum()

    return TN


def calc_fn(y_truth, y_pred):
    fn_list = pd.Series([1 if ((x[0] == 1) & (x[1] == -1)) else 0 for x in zip(y_truth, y_pred)])
    FN = fn_list.sum()

    return FN


def calc_stats(sag_id, level, include, gam, n, TP, FP, TN, FN, y_truth, y_pred):
    precision = TP / (TP + FP)
    sensitivity = TP / (TP + FN)
    N = TN + TP + FN + FP
    S = (TP + FN) / N
    P = (TP + FP) / N
    D = ((S * P) * (1 - S) * (1 - P)) ** (1 / 2)
    if D == 0:
        D = 1
    MCC = ((TP / N) - S * P) / D
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    oc_precision, oc_recall, _ = precision_recall_curve(y_truth, y_pred)
    AUC = auc(oc_recall, oc_precision)
    stat_list = [sag_id, level, include, gam, n, precision, sensitivity, MCC, AUC, F1,
                 N, S, P, TP, FP, TN, FN
                 ]

    return stat_list


def recruitSubs(p):
    abr_path, sag_id, minhash_sag_df, mg_covm_out, gamma, nu, src2contig_list, src2strain_list = p
    minhash_filter_df = minhash_sag_df.loc[(minhash_sag_df['jacc_sim_max'] == 1.0)]
    if len(minhash_filter_df['sag_id']) != 0:
        mg_covm_df = pd.read_csv(mg_covm_out, header=0, sep='\t', index_col=['contigName'])
        mg_covm_df.drop(columns=['contigLen', 'totalAvgDepth'], inplace=True)
        scale = StandardScaler().fit(mg_covm_df.values)
        scaled_data = scale.transform(mg_covm_df.values)
        std_merge_df = pd.DataFrame(scaled_data, index=mg_covm_df.index)
        recruit_contigs_df = std_merge_df.loc[std_merge_df.index.isin(
            list(minhash_filter_df['subcontig_id']))
        ]
        nonrecruit_filter_df = std_merge_df.copy()

        kmeans_pass_list = runKMEANS(recruit_contigs_df, sag_id, std_merge_df)
        kmeans_pass_df = pd.DataFrame(kmeans_pass_list,
                                      columns=['sag_id', 'subcontig_id', 'contig_id']
                                      )
        nonrecruit_kmeans_df = nonrecruit_filter_df.loc[nonrecruit_filter_df.index.isin(
            kmeans_pass_df['subcontig_id']
        )]
        final_pass_df = runOCSVM(recruit_contigs_df, nonrecruit_kmeans_df, sag_id, gamma, nu)
        # final_pass_df = pd.DataFrame(final_pass_list,
        #                             columns=['sag_id', 'nu', 'gamma', 'subcontig_id', 'contig_id', 'pred']
        #                             )

        complete_df = pd.DataFrame(nonrecruit_filter_df.index.values, columns=['subcontig_id'])
        complete_df['sag_id'] = sag_id
        complete_df['nu'] = nu
        complete_df['gamma'] = gamma
        complete_df['contig_id'] = [x.rsplit('_', 1)[0] for x in nonrecruit_filter_df.index.values]
        merge_recruits_df = pd.merge(complete_df, final_pass_df,
                                     on=['sag_id', 'nu', 'gamma', 'subcontig_id', 'contig_id'],
                                     how='outer'
                                     )
        merge_recruits_df.fillna(-1, inplace=True)
        merge_recruits_df['exact_truth'] = [1 if x in src2contig_list else -1
                                            for x in merge_recruits_df['contig_id']
                                            ]
        merge_recruits_df['strain_truth'] = [1 if x in src2strain_list else -1
                                             for x in merge_recruits_df['contig_id']
                                             ]
        subcontig_id_list = list(merge_recruits_df['subcontig_id'])
        contig_id_list = list(merge_recruits_df['contig_id'])
        exact_truth = list(merge_recruits_df['exact_truth'])
        strain_truth = list(merge_recruits_df['strain_truth'])
        pred = list(merge_recruits_df['pred'])
        stats_lists = recruit_stats([sag_id, gamma, nu, subcontig_id_list, contig_id_list,
                                     exact_truth, strain_truth, pred
                                     ])
        return stats_lists
    else:
        return [], [], [], []


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
    cat_clust_df = subset_clust_df.copy()  # pd.concat(block_list)
    std_id_df = pd.DataFrame(std_merge_df.index.values, columns=['subcontig_id'])
    std_id_df['contig_id'] = [x.rsplit('_', 1)[0] for x in std_id_df['subcontig_id']]
    cat_clust_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cat_clust_df['subcontig_id']]
    sub_std_df = std_id_df.loc[std_id_df['contig_id'].isin(list(cat_clust_df['contig_id']))]
    std_clust_df = sub_std_df.merge(cat_clust_df, on=['subcontig_id', 'contig_id'], how='outer')
    std_clust_df.fillna(-1, inplace=True)
    pred_df = std_clust_df[['subcontig_id', 'contig_id', 'kmeans_pred']]
    val_perc = pred_df.groupby('contig_id')['kmeans_pred'].value_counts(normalize=True).reset_index(name='percent')
    pos_perc = val_perc.loc[val_perc['kmeans_pred'] == 1]
    major_df = pos_perc.loc[pos_perc['percent'] >= 0.51]
    major_pred_df = pred_df.loc[pred_df['contig_id'].isin(major_df['contig_id'])]
    kmeans_pass_list = []
    for md_nm in major_pred_df['subcontig_id']:
        kmeans_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
    return kmeans_pass_list


def runOCSVM(sag_df, mg_df, sag_id, gamma, nu):
    # fit OCSVM
    clf = svm.OneClassSVM(nu=nu, gamma=gamma)
    clf.fit(sag_df.values)
    mg_pred = clf.predict(mg_df.values)
    contig_id_list = [x.rsplit('_', 1)[0] for x in mg_df.index.values]
    pred_df = pd.DataFrame(zip(mg_df.index.values, contig_id_list, mg_pred),
                           columns=['subcontig_id', 'contig_id', 'pred']
                           )
    pred_df['nu'] = nu
    pred_df['gamma'] = gamma
    pred_df['sag_id'] = sag_id
    pred_df = pred_df[['sag_id', 'nu', 'gamma', 'subcontig_id', 'contig_id', 'pred']]
    '''
    val_perc = pred_df.groupby('contig_id')['ocsvm_pred'].value_counts(
        normalize=True).reset_index(name='percent')
    pos_perc = val_perc.loc[val_perc['ocsvm_pred'] == 1]
    major_df = pos_perc.loc[pos_perc['percent'] >= 0.01]
    major_pred_df = pred_df.loc[pred_df['contig_id'].isin(major_df['contig_id'])]
    svm_pass_list = []
    for md_nm in major_pred_df['subcontig_id']:
        svm_pass_list.append([sag_id, nu, gamma, md_nm, md_nm.rsplit('_', 1)[0]])
    return svm_pass_list
    '''
    return pred_df


minhash_file = sys.argv[1]
mg_abund_file = sys.argv[2]
src2sag_file = sys.argv[3]

sag_id = sys.argv[4]
score_file_out = sys.argv[5]
pred_file_out = sys.argv[6]
nthreads = int(sys.argv[7])

minhash_df = pd.read_csv(minhash_file, header=0, sep='\t')

src2sag_df = pd.read_csv(src2sag_file, header=0, sep='\t')
src2sag_df = src2sag_df[src2sag_df['CAMI_genomeID'].notna()]
sag2src_dict = {}
sag2strain_dict = {}
for src_id in set(src2sag_df['CAMI_genomeID']):
    if src_id in sag_id:
        if sag_id in sag2src_dict.keys():
            if len(src_id) > len(sag2src_dict[sag_id]):
                sag2src_dict[sag_id] = src_id
                strain_id = list(src2sag_df.loc[src2sag_df['CAMI_genomeID'] == src_id]['strain'])[0]
                sag2strain_dict[sag_id] = strain_id

        else:
            sag2src_dict[sag_id] = src_id
            strain_id = list(src2sag_df.loc[src2sag_df['CAMI_genomeID'] == src_id]['strain'])[0]
            sag2strain_dict[sag_id] = strain_id
src2contig_df = src2sag_df.loc[src2sag_df['CAMI_genomeID'] == sag2src_dict[sag_id]]
src2strain_df = src2sag_df.loc[src2sag_df['strain'] == sag2strain_dict[sag_id]]
src2contig_list = list(set(src2contig_df['@@SEQUENCEID'].values))
src2strain_list = list(set(src2strain_df['@@SEQUENCEID'].values))

minhash_sag_df = minhash_df.loc[(minhash_df['sag_id'] == sag_id)]

if minhash_sag_df.shape[0] != 0:
    gamma_range = [10 ** k for k in range(-6, 6)]
    gamma_range.extend(['scale'])
    nu_range = [k / 10 for k in range(1, 10, 1)]
    ####
    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for gam in gamma_range:
        for n in nu_range:
            arg_list.append(['./abund/', sag_id, minhash_sag_df, mg_abund_file, gam, n,
                             src2contig_list, src2strain_list
                             ])
    results = pool.imap_unordered(recruitSubs, arg_list)
    score_list = []
    for i, output in enumerate(results, 1):
        print('\rRecruiting with Abundance Model: {}/{}'.format(i, len(arg_list)))
        score_list.append(output[0])
        score_list.append(output[1])
        score_list.append(output[2])
        score_list.append(output[3])

    logging.info('\n')
    pool.close()
    pool.join()
    score_df = pd.DataFrame(score_list, columns=['sag_id', 'level', 'inclusion', 'gamma', 'nu',
                                                 'precision', 'sensitivity', 'MCC', 'AUC', 'F1',
                                                 'N', 'S', 'P', 'TP', 'FP', 'TN', 'FN'
                                                 ])
    score_df.to_csv(score_file_out, index=False, sep='\t')
    sort_score_df = score_df.sort_values(['MCC'], ascending=[False])
    best_MCC = sort_score_df['MCC'].iloc[0]
    best_df = score_df.loc[score_df['MCC'] == best_MCC]
    '''
    pred_concat_df = pd.concat(pred_list)
    best_pred_list = []
    for i, row in best_df.iterrows():
        best_sagid = row['sag_id']
        best_gamma = row['gamma']
        best_nu = row['nu']
        best_trans = 'scaled'  # row['transformation']
        tmp_df = pred_concat_df.loc[((pred_concat_df['sag_id'] == best_sagid) &
                                     (pred_concat_df['gamma'] == best_gamma) &
                                     (pred_concat_df['nu'] == best_nu) &
                                     (pred_concat_df['transformation'] == best_trans)
                                     )]
        tmp_df = tmp_df[['sag_id', 'contig_id']]
        tmp_df = tmp_df.drop_duplicates(subset=['sag_id', 'contig_id'], keep='first')
        best_pred_list.append(tmp_df)
    pred_best_df = pd.concat(best_pred_list)
    pred_best_df.to_csv(pred_file_out, index=False, sep='\t')
    '''
else:
    print(sag_id, ' has no minhash recruits...')
