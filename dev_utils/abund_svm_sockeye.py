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


def predict_recruits(p):
    sag_id, gam, n, t_key, X_train, X_test, ocsvm, contig_id_list, y_test, y_test_strain = p
    ocsvm.fit(X_train)
    ocsvm_pred = ocsvm.predict(X_test)
    pred_df = pd.DataFrame(zip(X_test.index.values, contig_id_list, ocsvm_pred),
                           columns=['subcontig_id', 'contig_id', 'pred']
                           )
    pred_df['sag_id'] = sag_id
    pred_df['gamma'] = gam
    pred_df['nu'] = n
    pred_df['transformation'] = t_key

    pred_df = pred_df[['sag_id', 'gamma', 'nu', 'transformation', 'subcontig_id', 'contig_id', 'pred']]

    key_cnts = pred_df.groupby('contig_id')['pred'].count().reset_index()
    val_perc = pred_df.groupby('contig_id')['pred'].value_counts(
        normalize=True).reset_index(name='precent')
    pos_perc = val_perc.loc[val_perc['pred'] == 1]
    major_df = pos_perc.loc[pos_perc['precent'] >= 0.51]
    major_pred = [1 if x in list(major_df['contig_id']) else -1
                  for x in contig_id_list
                  ]
    pos_pred_list = list(set(pred_df.loc[pred_df['pred'] == 1]['contig_id']))
    all_pred = [1 if x in pos_pred_list else -1
                for x in contig_id_list
                ]
    pred_df['all_pred'] = all_pred
    pred_df['major_pred'] = major_pred
    pred_df['truth'] = y_test
    pred_df['truth_strain'] = y_test_strain
    # ALL Recruits
    # calculate for hybrid exact/strain-level matches
    TP = calc_tp(pred_df['truth'], pred_df['all_pred'])
    FP = calc_fp(pred_df['truth_strain'], pred_df['all_pred'])
    TN = calc_tn(pred_df['truth'], pred_df['all_pred'])
    FN = calc_fn(pred_df['truth'], pred_df['all_pred'])
    all_str_list = calc_stats(sag_id, 'strain', 'all', gam, n, t_key, TP, FP, TN, FN,
                              pred_df['truth_strain'], pred_df['all_pred']
                              )
    # ALL Recruits
    # calculate for exact-level match
    TP = calc_tp(pred_df['truth'], pred_df['all_pred'])
    FP = calc_fp(pred_df['truth'], pred_df['all_pred'])
    TN = calc_tn(pred_df['truth'], pred_df['all_pred'])
    FN = calc_fn(pred_df['truth'], pred_df['all_pred'])
    all_x_list = calc_stats(sag_id, 'exact', 'all', gam, n, t_key, TP, FP, TN, FN,
                            pred_df['truth'], pred_df['all_pred']
                            )

    # Majority-Rule Recruits
    # calculate for hybrid exact/strain-level matches
    TP = calc_tp(pred_df['truth'], pred_df['major_pred'])
    FP = calc_fp(pred_df['truth_strain'], pred_df['major_pred'])
    TN = calc_tn(pred_df['truth'], pred_df['major_pred'])
    FN = calc_fn(pred_df['truth'], pred_df['major_pred'])
    maj_str_list = calc_stats(sag_id, 'strain', 'majority', gam, n, t_key, TP, FP, TN, FN,
                              pred_df['truth_strain'], pred_df['major_pred']
                              )
    # Majority-Rule Recruits
    # calculate for exact-level match
    TP = calc_tp(pred_df['truth'], pred_df['major_pred'])
    FP = calc_fp(pred_df['truth'], pred_df['major_pred'])
    TN = calc_tn(pred_df['truth'], pred_df['major_pred'])
    FN = calc_fn(pred_df['truth'], pred_df['major_pred'])
    maj_x_list = calc_stats(sag_id, 'exact', 'majority', gam, n, t_key, TP, FP, TN, FN,
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


def calc_stats(sag_id, level, include, gam, n, t_key, TP, FP, TN, FN, y_truth, y_pred):
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
    stat_list = [sag_id, level, include, gam, n, t_key, precision, sensitivity, MCC, AUC, F1,
                 N, S, P, TP, FP, TN, FN
                 ]

    return stat_list


minhash_file = sys.argv[1]
mg_abund_file = sys.argv[2]
src2sag_file = sys.argv[3]

sag_id = sys.argv[4]
score_file_out = sys.argv[5]
pred_file_out = sys.argv[6]
nthreads = int(sys.argv[7])

minhash_df = pd.read_csv(minhash_file, header=0, sep='\t')
mg_abund_df = pd.read_csv(mg_abund_file, header=0, sep='\t', index_col=['contigName'])
del mg_abund_df['contigLen']
del mg_abund_df['totalAvgDepth']

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

grid_df_list = []
minhash_sag_df = minhash_df.loc[(minhash_df['sag_id'] == sag_id)]

if minhash_sag_df.shape[0] != 0:
    sag_merge_df = mg_abund_df.loc[mg_abund_df.index.isin(minhash_sag_df['subcontig_id'])]
    # norm = MinMaxScaler().fit(mg_abund_df.values)
    # normed_data = norm.transform(mg_abund_df.values)
    # norm_merge_df = pd.DataFrame(normed_data, index=mg_abund_df.index)
    scale = StandardScaler().fit(mg_abund_df.values)
    scaled_data = scale.transform(mg_abund_df.values)
    std_merge_df = pd.DataFrame(scaled_data, index=mg_abund_df.index)

    src2contig_df = src2sag_df.loc[src2sag_df['CAMI_genomeID'] == sag2src_dict[sag_id]]
    src2strain_df = src2sag_df.loc[src2sag_df['strain'] == sag2strain_dict[sag_id]]
    src2contig_list = list(set(src2contig_df['@@SEQUENCEID'].values))
    src2strain_list = list(set(src2strain_df['@@SEQUENCEID'].values))

    sag_cnt_list = [1 if src2contig_list.count(x.rsplit('_', 1)[0]) != 0 else -1
                    for x in sag_merge_df.index.values
                    ]
    sag_str_cnt_list = [1 if src2strain_list.count(x.rsplit('_', 1)[0]) != 0 else -1
                        for x in sag_merge_df.index.values
                        ]
    X_train_raw = sag_merge_df.values
    y_train = [1 for x in sag_merge_df.index.values]
    X_test_df = mg_abund_df.loc[~mg_abund_df.index.isin(sag_merge_df.index)]
    X_test_raw = X_test_df.values
    y_test = [1 if src2contig_list.count(x.rsplit('_', 1)[0]) != 0 else -1
              for x in X_test_df.index.values
              ]
    y_test_strain = [1 if src2strain_list.count(x.rsplit('_', 1)[0]) != 0 else -1
                     for x in X_test_df.index.values
                     ]

    # X_train_norm = norm_merge_df.loc[norm_merge_df.index.isin(sag_merge_df.index)]
    X_train_std = std_merge_df.loc[std_merge_df.index.isin(sag_merge_df.index)]
    # X_test_norm = norm_merge_df.loc[~norm_merge_df.index.isin(sag_merge_df.index)]
    X_test_std = std_merge_df.loc[~std_merge_df.index.isin(sag_merge_df.index)]

    contig_id_list = [x.rsplit('_', 1)[0] for x in X_test_std.index.values]
    gamma_range = [10 ** k for k in range(-6, 6)]
    gamma_range.extend(['scale'])
    nu_range = [k / 10 for k in range(1, 10, 1)]
    score_list = []
    print('Running data on OCSVM')
    # train_dict = {'raw': [X_train_raw, X_test_raw], 'normalized': [X_train_norm, X_test_norm],
    #              'scaled': [X_train_std, X_test_std]}
    train_dict = {'scaled': [X_train_std, X_test_std]}

    ####
    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for gam in gamma_range:
        for n in nu_range:
            for t_key in train_dict.keys():
                X_train = train_dict[t_key][0]
                X_test = train_dict[t_key][1]
                ocsvm = svm.OneClassSVM(gamma=gam, nu=n)
                arg_list.append([sag_id, gam, n, t_key, X_train, X_test, ocsvm, contig_id_list,
                                 y_test, y_test_strain])

    results = pool.imap_unordered(predict_recruits, arg_list)
    pred_list = []
    for i, output in enumerate(results):
        score_list.append(output[0])
        score_list.append(output[1])
        score_list.append(output[2])
        score_list.append(output[3])
        pred_list.append(output[4])
        sys.stderr.write('\rdone {}/{}'.format(i, len(arg_list)))
    pool.close()
    pool.join()
    ####

    score_df = pd.DataFrame(score_list, columns=['sag_id', 'level', 'inclusion', 'gamma', 'nu',
                                                 'transformation', 'precision', 'sensitivity', 'MCC',
                                                 'AUC', 'F1', 'N', 'S', 'P', 'TP', 'FP', 'TN', 'FN'
                                                 ])
    score_df.to_csv(score_file_out, index=False, sep='\t')
    sort_score_df = score_df.sort_values(['MCC'], ascending=[False])
    best_MCC = sort_score_df['MCC'].iloc[0]
    best_df = score_df.loc[score_df['MCC'] == best_MCC]

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

else:
    print(sag_id, ' has no minhash recruits...')
