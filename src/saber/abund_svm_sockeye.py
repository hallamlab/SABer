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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import multiprocessing


def predict_recruits(p):
    sag_id, gam, n, t_key, X_train, X_test, ocsvm, contig_id_list, y_test, y_test_strain = p
    ocsvm.fit(X_train)
    ocsvm_pred = ocsvm.predict(X_test)
    pred_df = pd.DataFrame(zip(X_test_df.index.values, contig_id_list, ocsvm_pred),
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
mg_merge_df = mg_abund_df
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
    sag_merge_df = mg_merge_df.loc[mg_merge_df.index.isin(minhash_sag_df['subcontig_id'])]
    norm = MinMaxScaler().fit(mg_merge_df.values)
    normed_data = norm.transform(mg_merge_df.values)
    norm_merge_df = pd.DataFrame(normed_data, index=mg_merge_df.index)
    scale = StandardScaler().fit(mg_merge_df.values)
    scaled_data = scale.transform(mg_merge_df.values)
    std_merge_df = pd.DataFrame(scaled_data, index=mg_merge_df.index)

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
    # print('Transforming data with SONG')
    # song_model = SONG(n_components=3, min_dist=0, n_neighbors=1)
    # song_trans = song_model.fit_transform(mg_tetra_df.values)
    # song_df = pd.DataFrame(song_trans, index=mg_tetra_df.index.values)
    # song_df.reset_index().to_csv('CAMI_high_GoldStandardAssembly.abund.song.tsv', sep='\t', index=False)
    # sag_song_df = song_df.loc[song_df.index.isin(minhash_sag_df['subcontig_id'])]
    # mg_song_df = song_df.loc[~song_df.index.isin(sag_tetra_df.index)]
    # song_train = sag_song_df.values
    # song_test = mg_song_df.values
    X_train_raw = sag_merge_df.values
    y_train = [1 for x in sag_merge_df.index.values]
    X_test_df = mg_merge_df.loc[~mg_merge_df.index.isin(sag_merge_df.index)]
    X_test_raw = X_test_df.values
    y_test = [1 if src2contig_list.count(x.rsplit('_', 1)[0]) != 0 else -1
              for x in X_test_df.index.values
              ]
    y_test_strain = [1 if src2strain_list.count(x.rsplit('_', 1)[0]) != 0 else -1
                     for x in X_test_df.index.values
                     ]

    X_train_norm = norm_merge_df.loc[norm_merge_df.index.isin(sag_merge_df.index)]
    X_train_std = std_merge_df.loc[std_merge_df.index.isin(sag_merge_df.index)]
    X_test_norm = norm_merge_df.loc[~norm_merge_df.index.isin(sag_merge_df.index)]
    X_test_std = std_merge_df.loc[~std_merge_df.index.isin(sag_merge_df.index)]

    contig_id_list = [x.rsplit('_', 1)[0] for x in X_test_df.index.values]
    gamma_range = [10 ** k for k in range(-6, 6)]
    gamma_range.extend(['scale'])
    nu_range = [k / 10 for k in range(1, 10, 1)]
    score_list = []
    print('Running untransformed data on OCSVM')
    train_dict = {'raw': [X_train_raw, X_test_raw], 'normalized': [X_train_norm, X_test_norm],
                  'scaled': [X_train_std, X_test_std]}
    ####
    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    gamma_range = [10]
    nu_range = [0.4]
    train_dict = {'scaled': [X_train_std, X_test_std]}
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
    '''
    for gam in gamma_range:
        for n in nu_range:
            for t_key in train_dict.keys():
                X_train = train_dict[t_key][0]
                X_test = train_dict[t_key][1]
                ocsvm = svm.OneClassSVM(gamma=gam, nu=n)
                ocsvm.fit(X_train)
                ocsvm_pred = ocsvm.predict(X_test)
                key = t_key + '_' + str(gam) + '_' + str(n)
                pred_df = pd.DataFrame(zip(X_test_df.index.values, contig_id_list, ocsvm_pred),
                                        columns=['subcontig_id', 'contig_id', key]
                                        )
                #key_cnts = pred_df.groupby('contig_id')[key].count().reset_index()

                #val_perc = pred_df.groupby('contig_id')[key].value_counts(
                #                                        normalize=True).reset_index(name='precent')
                
                #pos_perc = val_perc.loc[val_perc[key] == 1]
                
                #major_df = pos_perc.loc[pos_perc['precent'] >= 0.51]
                
                #updated_pred = [1 if x in list(major_df['contig_id']) else -1
                #                for x in contig_id_list
                #                ]
                pos_pred_list = list(set(pred_df.loc[pred_df[key] == 1]['contig_id']))
                updated_pred = [1 if x in pos_pred_list else -1
                                for x in contig_id_list
                                ]
                pred_df['updated_pred'] = updated_pred
                pred_df['truth'] = y_test
                pred_df['truth_strain'] = y_test_strain

                pred_df[key + '_tp'] = [1 if ((x[0]==1) & (x[1]==1)) else 0 for x in
                                       zip(pred_df['truth'], pred_df['updated_pred'])
                                       ]
                pred_df[key + '_fp'] = [1 if ((x[0]==-1) & (x[1]==1)) else 0 for x in
                                       zip(pred_df['truth_strain'], pred_df['updated_pred'])
                                       ]
                pred_df[key + '_tn'] = [1 if ((x[0]==-1) & (x[1]==-1)) else 0 for x in
                                       zip(pred_df['truth'], pred_df['updated_pred'])
                                       ]
                pred_df[key + '_fn'] = [1 if ((x[0]==1) & (x[1]==-1)) else 0 for x in
                                       zip(pred_df['truth'], pred_df['updated_pred'])
                                       ]
                pred_df['gamma'] = gam
                pred_df['nu'] = n

                TP = pred_df[key + '_tp'].sum()
                FP = pred_df[key + '_fp'].sum()
                TN = pred_df[key + '_tn'].sum()
                FN = pred_df[key + '_fn'].sum()
                precision = TP / (TP + FP)
                sensitivity = TP / (TP + FN)
                N = TN + TP + FN + FP
                S = (TP + FN) / N
                P = (TP + FP) / N
                MCC = ((TP/N)-S*P) / ((S*P)*(1-S)*(1-P))**(1/2)
                F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
                oc_precision, oc_recall, _ = precision_recall_curve(y_test_strain, updated_pred)
                AUC = auc(oc_recall, oc_precision)
                print(t_key + ' data:')
                print('One-Class SVM: nu={}, gamma={}, AUC={:.4f}, MCC={:.4f}, F1={:.4f}'.format(
                        n, gam, AUC, MCC, F1)
                        )
                print('PR OCSVM vs No Skill: P={:.4f}, R={:.4f}, NS_P={:.4f}, NS_R={:.4f}'.format(
                        precision, sensitivity, P, S)
                        )
                score_list.append([sag_id, gam, n, t_key, precision, sensitivity, MCC, AUC, F1, N, S,
                                   P, TP, FP, TN, FN
                                   ])
        '''

    score_df = pd.DataFrame(score_list, columns=['sag_id', 'level', 'inclusion', 'gamma', 'nu',
                                                 'transformation', 'precision', 'sensitivity', 'MCC',
                                                 'AUC', 'F1', 'N', 'S', 'P', 'TP', 'FP', 'TN', 'FN'
                                                 ])
    # score_df.to_csv(score_file_out, index=False, sep='\t')
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

'''
    print('Transforming data with SONG')
    song_model = SONG(n_components=20, min_dist=0)
    song_model.fit(X_test)
    song_train = song_model.transform(X_train)
    song_test = song_model.transform(X_test)

    print('Running grid search with SONG + OCSVM')
    song_df_list = []
    for score in scores:
        ocsvm = svm.OneClassSVM()
        #song = SONG()
        #pipeline = Pipeline([("song", song), ("ocsvm", ocsvm)])
        pipeline = Pipeline([("ocsvm", ocsvm)])
        params_grid_pipeline = {
            "ocsvm__gamma": gamma_range,
            "ocsvm__nu": nu_range
            #"song__n_neighbors": [5, 20, 30],
            #"song__n_components": [10, 20, 30, 40],
            #"song__min_dist": [0],
            #"ocsvm__gamma": [best_gamma],
            #"ocsvm__nu": [best_nu]
            }

        clf_pipeline = GridSearchCV(pipeline, params_grid_pipeline, scoring='%s_macro' % score,
                                    return_train_score=True, n_jobs=8
                                    )
        clf_pipeline.fit(song_train, y_train)

        resultDf = pd.DataFrame(clf_pipeline.cv_results_)
        filtered_results_df = resultDf[["mean_test_score", "std_test_score", "params"]]
        filtered_results_df.columns = ['mean_' + score, 'std_' + score, 'params']
        filtered_results_df['sag_id'] = sag_id
        filtered_results_df.params.dropna().apply(pd.Series)
        result_params_df = filtered_results_df.drop('params', 1).assign(
                                **pd.DataFrame.from_records(
                                    filtered_results_df.params.dropna().tolist(),
                                    index = filtered_results_df.params.dropna().index)
                                    )
        song_df_list.append(result_params_df)
    on_list = ['sag_id', 'nu', 'gamma']
    merged_result_df = reduce(lambda  left,right: pd.merge(left,right,
                                        on=on_list,
                                        how='outer'), scores_df_list)
    sorted_results_df = merged_result_df.sort_values(by=["mean_precision", "mean_recall"],
                                                     ascending=[False, False]
                                                     )
    sorted_results_df = sorted_results_df[['sag_id', 'gamma', 'nu', 'mean_recall', 'std_recall',
                                           'mean_precision', 'std_precision'
                                           ]]
    best_nu = sorted_results_df['nu'].iloc[0]
    best_gamma = sorted_results_df['gamma'].iloc[0]
    best_params = {"gamma": [float(best_gamma)],
                   "nu": [float(best_nu)]
                   }
    print('Best gamma is {}, best nu is {}'.format(best_gamma, best_nu))
'''
'''
    on_list = ['sag_id', 'song__n_neighbors', 'song__n_components', 'song__min_dist',
               'ocsvm__gamma', 'ocsvm__nu'
                ]
    merged_result_df = reduce(lambda  left,right: pd.merge(left,right,
                                        on=on_list,
                                        how='outer'), song_df_list)
    sorted_results_df = merged_result_df.sort_values(by=["mean_precision", "mean_recall"],
                                                     ascending=[False, False]
                                                     )
    sorted_results_df = sorted_results_df[['sag_id', 'ocsvm__gamma', 'ocsvm__nu',
                                           'song__min_dist', 'song__n_components',
                                           'song__n_neighbors',
                                           'mean_recall', 'std_recall',
                                           'mean_precision', 'std_precision'
                                           ]]

    best_n_neighbors = sorted_results_df['song__n_neighbors'].iloc[0]
    best_n_components = sorted_results_df['song__n_components'].iloc[0]
    best_min_dist = sorted_results_df['song__min_dist'].iloc[0]
    best_params_grid_pipeline = {
            "song__n_neighbors": [best_n_neighbors],
            "song__n_components": [best_n_components],
            "song__min_dist": [best_min_dist],
            "ocsvm__gamma": [best_gamma],
            "ocsvm__nu": [best_nu]
            }
    print('Best n_neighbors is {}, best n_components is {}'.format(best_n_neighbors, best_n_components))
'''
'''
    print('Running SONG transformed data on OCSVM')
    for score in scores:
        ocsvm = svm.OneClassSVM()
        best_clf = GridSearchCV(ocsvm, best_params, scoring='%s_macro' % score)
        best_clf.fit(song_train, y_train)
        print("{} on the test set with SONG transformation: {:.3f}".format('%s_macro' % score,
                best_clf.score(song_test, y_test)
                ))
        score_dict['song_%s_macro' % score] = best_clf.score(song_test, y_test)
        song_pred = best_clf.predict(song_test)

    for key in score_dict.keys():
        sorted_results_df[key] = score_dict[key]
    sorted_results_df.to_csv(song_file_out, index=False, sep='\t')

    pred_df = pd.DataFrame(zip(mg_tetra_df.index.values, ocsvm_pred, song_pred),
                            columns = ['subcontig_id', 'ocsvm_pred', 'song_pred']
                            )
    
    #pred_df = pd.read_csv(pred_file_out, header=0, sep='\t')
    pred_df['truth'] = y_test
    pred_df['ocsvm_tp'] = [1 if ((x[0]==1) & (x[1]==1)) else 0 for x in
                           zip(pred_df['truth'], pred_df['ocsvm_pred'])
                           ]
    pred_df['ocsvm_fp'] = [1 if ((x[0]==-1) & (x[1]==1)) else 0 for x in
                           zip(pred_df['truth'], pred_df['ocsvm_pred'])
                           ]
    pred_df['ocsvm_tn'] = [1 if ((x[0]==-1) & (x[1]==-1)) else 0 for x in
                           zip(pred_df['truth'], pred_df['ocsvm_pred'])
                           ]
    pred_df['ocsvm_fn'] = [1 if ((x[0]==1) & (x[1]==-1)) else 0 for x in
                           zip(pred_df['truth'], pred_df['ocsvm_pred'])
                           ]


    pred_df['song_tp'] = [1 if ((x[0]==1) & (x[1]==1)) else 0 for x in
                           zip(pred_df['truth'], pred_df['song_pred'])
                           ]
    pred_df['song_fp'] = [1 if ((x[0]==-1) & (x[1]==1)) else 0 for x in
                           zip(pred_df['truth'], pred_df['song_pred'])
                           ]
    pred_df['song_tn'] = [1 if ((x[0]==-1) & (x[1]==-1)) else 0 for x in
                           zip(pred_df['truth'], pred_df['song_pred'])
                           ]
    pred_df['song_fn'] = [1 if ((x[0]==1) & (x[1]==-1)) else 0 for x in
                           zip(pred_df['truth'], pred_df['song_pred'])
                           ]

    pred_df['ocsvm_precision'] = pred_df['ocsvm_tp'].sum() / \
                                    (pred_df['ocsvm_tp'].sum() + pred_df['ocsvm_fp'].sum())
    pred_df['ocsvm_sensitivity'] = pred_df['ocsvm_tp'].sum() / \
                                    (pred_df['ocsvm_tp'].sum() + pred_df['ocsvm_fn'].sum())
    pred_df['song_precision'] = pred_df['song_tp'].sum() / \
                                    (pred_df['song_tp'].sum() + pred_df['song_fp'].sum())
    pred_df['song_sensitivity'] = pred_df['ocsvm_tp'].sum() / \
                                    (pred_df['song_tp'].sum() + pred_df['song_fn'].sum())

    pred_df.to_csv(pred_file_out, index=False, sep='\t')
    



# END of script
sys.exit()
# Make a toy dataset
X, y = make_classification(
    n_samples=1000,
    n_features=300,
    n_informative=250,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=1212,
)

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Classification with a linear SVM
svc = LinearSVC(dual=False, random_state=123)
params_grid = {"C": [10 ** k for k in range(-3, 4)]}
clf = GridSearchCV(svc, params_grid)
clf.fit(X_train, y_train)
print(
    "Accuracy on the test set with raw data: {:.3f}".format(clf.score(X_test, y_test))
)

# Transformation with UMAP followed by classification with a linear SVM
umap = UMAP(random_state=456)
pipeline = Pipeline([("umap", umap), ("svc", svc)])
params_grid_pipeline = {
    "umap__n_neighbors": [5, 20],
    "umap__n_components": [15, 25, 50],
    "umap__min_dist": [0],
    "svc__C": [10 ** k for k in range(-3, 4)],
}


clf_pipeline = GridSearchCV(pipeline, params_grid_pipeline)
clf_pipeline.fit(X_train, y_train)
print(
    "Accuracy on the test set with UMAP transformation: {:.3f}".format(
        clf_pipeline.score(X_test, y_test)
    )
)
print(clf_pipeline.best_estimator_)
print(clf_pipeline.best_score_)
print(clf_pipeline.best_params_)

'''
