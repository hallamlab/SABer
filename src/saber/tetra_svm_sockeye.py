from sklearn import svm
import sys
import pandas as pd
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc #, f1_score, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import multiprocessing



def predict_recruits(p):
    sag_id, gam, n, t_key, X_train, X_test, ocsvm, contig_id_list, y_test, y_test_strain = p
    ocsvm.fit(X_train)
    ocsvm_pred = ocsvm.predict(X_test)
    key = t_key + '_' + str(gam) + '_' + str(n)
    pred_df = pd.DataFrame(zip(X_test_df.index.values, contig_id_list, ocsvm_pred),
                            columns=['subcontig_id', 'contig_id', key]
                            )
    
    key_cnts = pred_df.groupby('contig_id')[key].count().reset_index()
    val_perc = pred_df.groupby('contig_id')[key].value_counts(
                                            normalize=True).reset_index(name='precent')
    pos_perc = val_perc.loc[val_perc[key] == 1]
    major_df = pos_perc.loc[pos_perc['precent'] >= 0.51]
    major_pred = [1 if x in list(major_df['contig_id']) else -1
                    for x in contig_id_list
                    ]
    pos_pred_list = list(set(pred_df.loc[pred_df[key] == 1]['contig_id']))
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


    return all_str_list, all_x_list, maj_str_list, maj_x_list


def calc_tp(y_truth, y_pred):
    tp_list = pd.Series([1 if ((x[0]==1) & (x[1]==1)) else 0 for x in zip(y_truth, y_pred)])
    TP = tp_list.sum()

    return TP


def calc_fp(y_truth, y_pred):
    fp_list = pd.Series([1 if ((x[0]==-1) & (x[1]==1)) else 0 for x in zip(y_truth, y_pred)])
    FP = fp_list.sum()

    return FP


def calc_tn(y_truth, y_pred):
    tn_list = pd.Series([1 if ((x[0]==-1) & (x[1]==-1)) else 0 for x in zip(y_truth, y_pred)])
    TN = tn_list.sum()

    return TN


def calc_fn(y_truth, y_pred):
    fn_list = pd.Series([1 if ((x[0]==1) & (x[1]==-1)) else 0 for x in zip(y_truth, y_pred)])
    FN = fn_list.sum()

    return FN


def calc_stats(sag_id, level, include, gam, n, t_key, TP, FP, TN, FN, y_truth, y_pred):
    precision = TP / (TP + FP)
    sensitivity = TP / (TP + FN)
    N = TN + TP + FN + FP
    S = (TP + FN) / N
    P = (TP + FP) / N
    D = ((S*P)*(1-S)*(1-P))**(1/2)
    if D == 0:
        D = 1
    MCC = ((TP/N)-S*P) / D
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    oc_precision, oc_recall, _ = precision_recall_curve(y_truth, y_pred)
    AUC = auc(oc_recall, oc_precision)
    stat_list = [sag_id, level, include, gam, n, t_key, precision, sensitivity, MCC, AUC, F1,
                 N, S, P, TP, FP, TN, FN
                 ]

    return stat_list




minhash_file = sys.argv[1]
mg_tetra_file = sys.argv[2]
abund_recruits_file = sys.argv[3]
src2sag_file = sys.argv[4]
sag_id_file = sys.argv[5]
#score_file_out = sys.argv[6]
nthreads = int(sys.argv[6])

minhash_df = pd.read_csv(minhash_file, header=0, sep='\t')
mg_tetra_df = pd.read_csv(mg_tetra_file, header=0, sep='\t', index_col=['index'])
abund_recruits_df = pd.read_csv(abund_recruits_file, header=0, sep='\t')
with open(sag_id_file, 'r') as f:
    sag_id_list = f.read().splitlines()
src2sag_df = pd.read_csv(src2sag_file, header=0, sep='\t')
src2sag_df = src2sag_df[src2sag_df['CAMI_genomeID'].notna()]

for sag_id in sag_id_list:
    print(sag_id)
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

    major_abund_df = abund_recruits_df.loc[abund_recruits_df['sag_id'] == sag_id]
    mg_tetra_df['contig_id'] = [x.rsplit('_', 1)[0] for x in mg_tetra_df.index]
    mg_abund_df = mg_tetra_df.loc[mg_tetra_df['contig_id'].isin(list(major_abund_df['contig_id']))]
    mg_abund_df.drop(columns=['contig_id'], inplace=True)
    mg_tetra_df.drop(columns=['contig_id'], inplace=True)
                
    minhash_sag_df = minhash_df.loc[(minhash_df['sag_id'] == sag_id) &
                                    (minhash_df['jacc_sim_max'] == 1.0)
                                    ]
    if minhash_sag_df.shape[0] != 0:
        sag_merge_df = mg_tetra_df.loc[mg_tetra_df.index.isin(minhash_sag_df['subcontig_id'])]
        norm = MinMaxScaler().fit(mg_tetra_df.values)
        normed_data = norm.transform(mg_tetra_df.values)
        norm_merge_df = pd.DataFrame(normed_data, index=mg_tetra_df.index)
        scale = StandardScaler().fit(mg_tetra_df.values)
        scaled_data = scale.transform(mg_tetra_df.values)
        std_merge_df = pd.DataFrame(scaled_data, index=mg_tetra_df.index)
        
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
        if X_test_df.shape[0] != 0:
            X_test_raw = X_test_df.values
            y_test = [1 if src2contig_list.count(x.rsplit('_', 1)[0]) != 0 else -1
                        for x in X_test_df.index.values
                        ]
            y_test_strain = [1 if src2strain_list.count(x.rsplit('_', 1)[0]) != 0 else -1
                             for x in X_test_df.index.values
                             ]

            X_train_norm = norm_merge_df.loc[norm_merge_df.index.isin(sag_merge_df.index)]
            X_train_std = std_merge_df.loc[std_merge_df.index.isin(sag_merge_df.index)]
            X_test_norm = norm_merge_df.loc[((~norm_merge_df.index.isin(sag_merge_df.index)) &
                                             (norm_merge_df.index.isin(list(mg_abund_df.index)))
                                             )]
            X_test_std = std_merge_df.loc[((~std_merge_df.index.isin(sag_merge_df.index)) &
                                           (std_merge_df.index.isin(list(mg_abund_df.index)))
                                           )]


            contig_id_list = [x.rsplit('_', 1)[0] for x in X_test_df.index.values]
            gamma_range = [10 ** k for k in range(-6, 6)]
            gamma_range.extend(['scale'])
            nu_range = [k/10 for k in range(1, 10, 1)]
            score_list = []
            print('Running data on OCSVM')
            train_dict = {'raw': [X_train_raw, X_test_raw], 'normalized': [X_train_norm, X_test_norm],
                          'scaled': [X_train_std, X_test_std]
                          }
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
            for i, output in enumerate(results):
                score_list.append(output[0])
                score_list.append(output[1])
                score_list.append(output[2])
                score_list.append(output[3])
                sys.stderr.write('\rdone {}/{}'.format(i, len(arg_list)))
            pool.close()
            pool.join()
            ####
            score_df = pd.DataFrame(score_list, columns=['sag_id', 'level', 'inclusion', 'gamma', 'nu',
                                                         'transformation', 'precision', 'sensitivity', 'MCC',
                                                         'AUC', 'F1', 'N', 'S', 'P', 'TP', 'FP', 'TN', 'FN'
                                                         ])
            score_file_out = 'tetra_song_20/' + sag_id + '.SCORES.tetra_song_20.tsv'
            score_df.to_csv(score_file_out, index=False, sep='\t')
        else:
            print(sag_id, ' has no abundance recruits...')

    else:
        print(sag_id, ' has no minhash recruits...')
