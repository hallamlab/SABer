from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm
from umap import UMAP
import sys
import pandas as pd
from functools import reduce
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from song.song import SONG




minhash_file = sys.argv[1]
mg_tetra_file = sys.argv[2]
src2sag_file = sys.argv[3]

sag_id = sys.argv[4]
ocsvm_file_out = sys.argv[5]
song_file_out = sys.argv[6]
pred_file_out = sys.argv[7]
score_file_out = sys.argv[8]

minhash_df = pd.read_csv(minhash_file, header=0, sep='\t')
mg_tetra_df = pd.read_csv(mg_tetra_file, header=0, sep='\t', index_col=['contigName'])
src2sag_df = pd.read_csv(src2sag_file, header=0, sep='\t')
src2sag_df = src2sag_df[src2sag_df['CAMI_genomeID'].notna()]
sag2src_dict = {}
sag2strain_dict = {}
for src_id in set(src2sag_df['CAMI_genomeID']):
    if src_id in sag_id:
        sag2src_dict[sag_id] = src_id
        strain_id = list(src2sag_df.loc[src2sag_df['CAMI_genomeID'] == src_id]['strain'])[0]
        sag2strain_dict[sag_id] = strain_id

grid_df_list = []
#for sag_id in list(set(minhash_df['sag_id'])):
minhash_sag_df = minhash_df.loc[(minhash_df['sag_id'] == sag_id) &
                                (minhash_df['jacc_sim'] == 1.0)
                                ]
if minhash_sag_df.shape[0] != 0:
    sag_mh_contig_list = list(set(minhash_sag_df['contig_id'].values))
    sag_tetra_contig_list = [x for x in mg_tetra_df.index.values
                     if sag_mh_contig_list.count(x.rsplit('_', 1)[0]) != 0
                     ]
    sag_tetra_df = mg_tetra_df.loc[mg_tetra_df.index.isin(sag_tetra_contig_list)]
    src2contig_df = src2sag_df.loc[src2sag_df['CAMI_genomeID'] == sag2src_dict[sag_id]]
    src2strain_df = src2sag_df.loc[src2sag_df['strain'] == sag2strain_dict[sag_id]]
    src2contig_list = list(set(src2contig_df['@@SEQUENCEID'].values))
    src2strain_list = list(set(src2contig_df['@@SEQUENCEID'].values))
    X_train = sag_tetra_df.values
    y_train = [1 for x in sag_tetra_df.index.values]
    X_test = mg_tetra_df.values
    y_test = [1 if src2contig_list.count(x.rsplit('_', 1)[0]) != 0 else -1
                for x in mg_tetra_df.index.values
                ]
    y_test_strain = [1 if src2strain_list.count(x.rsplit('_', 1)[0]) != 0 else -1
                     for x in mg_tetra_df.index.values
                     ]
    contig_id_list = [x.rsplit('_', 1)[0] for x in mg_tetra_df.index.values]
    #truth_df = pd.DataFrame(zip(contig_id_list, y_test, y_test_strain),
    #                        columns=['contig_id', 'truth', 'truth_strain']
    #                        ).drop_duplicates()
    gamma_range = [10 ** k for k in range(-6, 6)]
    gamma_range.extend(['scale'])
    nu_range = [k/10 for k in range(1, 10, 1)]
    scores = ['precision', 'recall']
    '''
    print('Running grid search with OCSVM only')
    scores_df_list = []
    for score in scores:
        ocsvm = svm.OneClassSVM()
        params_grid = {"gamma": gamma_range,
                       "nu": nu_range
                       }
        clf = GridSearchCV(ocsvm, params_grid, scoring='%s_weighted' % score,
                           return_train_score=True, n_jobs=8
                           )
        clf.fit(X_train, y_train)

        resultDf = pd.DataFrame(clf.cv_results_)
        filtered_results_df = resultDf[["mean_test_score", "std_test_score", "params"]]
        filtered_results_df.columns = ['mean_' + score, 'std_' + score, 'params']
        filtered_results_df['sag_id'] = sag_id
        filtered_results_df.params.dropna().apply(pd.Series)
        result_params_df = filtered_results_df.drop('params', 1).assign(
                                **pd.DataFrame.from_records(
                                    filtered_results_df.params.dropna().tolist(),
                                    index = filtered_results_df.params.dropna().index)
                                    )
        scores_df_list.append(result_params_df)

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
    sorted_results_df.to_csv(ocsvm_file_out, index=False, sep='\t')
    #print(sag_id)
    #print(sorted_results_df.head())
    #grid_df_list.append(sorted_results_df)

    best_nu = sorted_results_df['nu'].iloc[0]
    best_gamma = sorted_results_df['gamma'].iloc[0]
    best_params = {"gamma": [float(best_gamma)],
                   "nu": [float(best_nu)]
                   }
    print('Best gamma is {}, best nu is {}'.format(best_gamma, best_nu))
    '''
    score_list = []
    print('Running untransformed data on OCSVM')
    for gam in gamma_range:
        for n in nu_range:
            ocsvm = svm.OneClassSVM(gamma=gam, nu=n)
            ocsvm.fit(X_train)
           
            key = str(gam) + '_' + str(n)

            ocsvm_pred = ocsvm.predict(X_test)
            pred_df = pd.DataFrame(zip(mg_tetra_df.index.values, contig_id_list, ocsvm_pred),
                                    columns=['subcontig_id', 'contig_id', key]
                                    )
            key_cnts = pred_df.groupby('contig_id')[key].count().reset_index()

            val_perc = pred_df.groupby('contig_id')[key].value_counts(normalize=True).reset_index(name='precent')
            pos_perc = val_perc.loc[val_perc[key] == 1]
            major_df = pos_perc.loc[pos_perc['precent'] >= 0.51]
            
            updated_pred = [1 if x[0] in list(major_df['contig_id']) else x[1]
                             for x in zip(contig_id_list, ocsvm_pred)
                             ]

            pred_df = pd.DataFrame(zip(mg_tetra_df.index.values, updated_pred),
                                    columns = ['subcontig_id', key]
                                    )
            pred_df['truth'] = y_test
            pred_df['truth_strain'] = y_test_strain
            pred_df[key + '_tp'] = [1 if ((x[0]==1) & (x[1]==1)) else 0 for x in
                                   zip(pred_df['truth'], pred_df[key])
                                   ]
            pred_df[key + '_fp'] = [1 if ((x[0]==-1) & (x[1]==1)) else 0 for x in
                                   zip(pred_df['truth_strain'], pred_df[key])
                                   ]
            pred_df[key + '_tn'] = [1 if ((x[0]==-1) & (x[1]==-1)) else 0 for x in
                                   zip(pred_df['truth'], pred_df[key])
                                   ]
            pred_df[key + '_fn'] = [1 if ((x[0]==1) & (x[1]==-1)) else 0 for x in
                                   zip(pred_df['truth'], pred_df[key])
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
            score_list.append([sag_id, gam, n, precision, sensitivity, MCC, N, S, P, TP, FP, TN, FN])
            print(key, precision, sensitivity, MCC)
    
    score_df = pd.DataFrame(score_list, columns=['sag_id', 'gamma', 'nu', 'precision', 'sensitivity',
                                                 'MCC', 'N', 'S', 'P', 'TP', 'FP', 'TN', 'FN'
                                                 ])
    score_df.to_csv(score_file_out, index=False, sep='\t')







    sys.exit()
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

