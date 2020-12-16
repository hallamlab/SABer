import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import numpy as np
import sys
import os
from os import listdir, makedirs, path
pd.set_option('display.max_columns', None)
from sklearn.metrics import auc
import math
pd.set_option('mode.chained_assignment', None)
import itertools

sns.set_context("poster")
sns.set_style('whitegrid')
sns.set(font_scale=1.0)


pred_file = 'CAMI_high_GoldStandardAssembly.SCORES.filtered.tsv'
src2sag_file = 'src2sag_map.tsv'
abund_file = 'CAMI_high_GoldStandardAssembly.metabat.trimmed.tsv'

pred_df = pd.read_csv(pred_file, header=0, sep='\t')
trans_map = {'normalized': 'MinMaxScaler', 'scaled': 'StandardScaler', 'raw': 'raw'}
pred_df['transformation'] = [trans_map[x] for x in pred_df['transformation']]
src2sag_df = pd.read_csv(src2sag_file, header=0, sep='\t')
abund_df = pd.read_csv(abund_file, header=0, sep='\t')

abund_df['contig_id'] = [x.rsplit('_', 1)[0] for x in abund_df['contigName']]
count_df = abund_df.groupby(['contig_id'])['contigName'].count().reset_index()

src2sag_df = src2sag_df[src2sag_df['CAMI_genomeID'].notna()]
src2sag_df = pd.merge(src2sag_df, count_df, left_on='@@SEQUENCEID', right_on='contig_id', how='left')
src_count_df = src2sag_df.groupby(['CAMI_genomeID'])['contigName'].sum().reset_index()
src_count_df.columns = ['CAMI_genomeID', 'subcontig_count']
sag2src_dict = {}
for sag_id in set(pred_df['sag_id']):
    for src_id in set(src2sag_df['CAMI_genomeID']):
        if src_id in sag_id:
            if sag_id in sag2src_dict.keys():
                if len(src_id) > len(sag2src_dict[sag_id]):
                    sag2src_dict[sag_id] = src_id

            else:
                sag2src_dict[sag_id] = src_id
pred_df['CAMI_genomeID'] = [sag2src_dict[x] for x in pred_df['sag_id']]
pred_df = pd.merge(pred_df, src_count_df, on='CAMI_genomeID', how='left')
pred_df = pred_df.dropna()
pred_df = pred_df.loc[((pred_df['TP'] > 0) | (pred_df['FP'] > 0))]
pred_df['nu_gamma'] = [str(x[0]) + '_' + str(x[1]) for x in 
                                zip(pred_df['nu'], pred_df['gamma'])
                                ]
pred_df = pred_df.sort_values(['MCC'], ascending=[False])
PR_df = pred_df.drop_duplicates(subset=['sag_id'], keep='first')
PR_df = PR_df[['sag_id', 'MCC']]
PR_df.columns = ['sag_id', 'best_MCC']
merge_df = pd.merge(pred_df, PR_df, on=['sag_id'])
filter_df = merge_df.loc[(merge_df['MCC'] >= merge_df['best_MCC'])]
incl_dict = {'majority': 0, 'all': 1}
lev_dict = {'strain': 0, 'exact': 1}
trans_dict = {'StandardScaler': 0, 'MinMaxScaler': 1, 'raw': 2}
gamma_dict = {'scale':0, '1e-06':1, '1e-05':2, '0.0001':3, '0.001':4, '0.01':5, '0.1':6,
              '1':7, '10':8, '100':9, '1000':10, '10000':11, '100000':12
              }
filter_df['inclusion_sorter'] = [incl_dict[x] for x in filter_df['inclusion']]
filter_df['level_sorter'] = [lev_dict[x] for x in filter_df['level']]
filter_df['transformation_sorter'] = [trans_dict[x] for x in filter_df['transformation']]
filter_df['gamma_sorter'] = [gamma_dict[x] for x in filter_df['gamma']]
filter_df = filter_df.sort_values(['nu', 'gamma_sorter', 'transformation_sorter',
                                   'inclusion_sorter', 'level_sorter'],
                                   ascending=[False, True, True, True, True]
                                   )
num_1_df = filter_df.drop_duplicates(subset=['sag_id'], keep='first')

keep_list = ['sag_id', 'level', 'inclusion', 'transformation', 'gamma', 'nu', 'precision', 'MCC', 'sensitivity']
pred_stack_df = num_1_df[keep_list].set_index(['sag_id', 'level', 'inclusion',
                                              'transformation', 'gamma', 'nu']).stack().reset_index()
pred_stack_df.columns = ['sag_id', 'level', 'inclusion', 'transformation',
                         'gamma', 'nu', 'metric', 'score'
                         ]
pred_stack_df['combo'] = [x[0] + '_' + x[1] + '_' + x[2] for x in zip(pred_stack_df['level'],
                                                                      pred_stack_df['inclusion'],
                                                                      pred_stack_df['transformation'])
                                                                       ]

mean_df = pred_stack_df.groupby(['level', 'inclusion', 'transformation', 'metric']
                                )['score'].mean().unstack('metric').reset_index()
mean_df['round_MCC'] = mean_df['MCC'].round(2)
mean_df = mean_df.sort_values(['round_MCC'], ascending=[False])
mean_df.to_csv('PR_plots/Mean_stats.tsv', sep='\t', index=False)

flierprops = dict(markerfacecolor='0.75', markersize=5, markeredgecolor='w',
              linestyle='none')

ax = sns.catplot(x="metric", y="score", hue="level", kind='box',
                    data=pred_stack_df, aspect=2, palette=sns.light_palette("black"),
                    flierprops=flierprops)
plt.savefig('PR_plots/level_boxplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

ax = sns.catplot(x="metric", y="score", hue="inclusion", kind='box',
                    data=pred_stack_df, aspect=2, palette=sns.light_palette("black"),
                    flierprops=flierprops)
plt.savefig('PR_plots/inclusion_boxplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

ax = sns.catplot(x="metric", y="score", hue="transformation", kind='box',
                    data=pred_stack_df, aspect=2, palette=sns.light_palette("black"),
                    flierprops=flierprops)
plt.savefig('PR_plots/transformation_boxplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

ax = sns.catplot(x="metric", y="score", hue="combo", kind='box',
                    data=pred_stack_df, aspect=2, palette=sns.light_palette("black"),
                    flierprops=flierprops)
plt.savefig('PR_plots/combo_boxplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

count_df = num_1_df[['level', 'inclusion', 'transformation', 'sag_id']
                         ].groupby(['level', 'inclusion', 'transformation']
                                   ).count().reset_index()
count_df.columns = ['level', 'inclusion', 'transformation', 'count']
count_df['combo'] = [x[0] + '_' + x[1] + '_' + x[2] for x in zip(count_df['level'],
                                                                      count_df['inclusion'],
                                                                      count_df['transformation'])
                                                                       ]

count_df = count_df.sort_values(['count'], ascending=[False])
g = sns.barplot(data=count_df, x="count", y="combo")
plt.savefig('PR_plots/combo_barplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

top_level = count_df['level'].iloc[0]
top_inclusion = count_df['inclusion'].iloc[0]
top_transformation = count_df['transformation'].iloc[0]
top_df = pred_df.loc[((pred_df['level'] == top_level) &
                       (pred_df['inclusion'] == top_inclusion) &
                       (pred_df['transformation'] == top_transformation))
                       ]

'''
PR_top_df = top_df.drop_duplicates(subset=['sag_id'],
                                   keep='first'
                                   )

PR_top_df = PR_top_df[['sag_id', 'precision', 'MCC']]
PR_top_df.columns = ['sag_id', 'best_precision', 'best_MCC']
merge_top_df = pd.merge(top_df, PR_top_df, on=['sag_id'], how='left')
filter_top_df = merge_top_df.loc[((merge_top_df['precision'] >= merge_top_df['best_precision']) & 
                          (merge_top_df['MCC'] >= merge_top_df['best_MCC'])
                          )]
filter_top_df = filter_top_df.loc[filter_top_df['gamma'] != 'scale']
filter_top_df['gamma'] = pd.to_numeric(filter_top_df['gamma'], errors='coerce')
filter_top_df = filter_top_df.sort_values(['nu', 'gamma'], ascending=[False, True])
num_1_df = filter_top_df.drop_duplicates(subset=['sag_id'], keep='first')
num_1_df['rank'] = (num_1_df['precision'] + num_1_df['MCC']
                    ).astype(float).rank(method='dense', ascending=False).astype(float)
'''
best_sub_list = []
rank_sub_list = []
for sag_id in set(top_df['sag_id']):
    sag_top_df = top_df.loc[top_df['sag_id'] == sag_id]
    sag_top_df['round_MCC'] = sag_top_df['MCC'].round(2)
    sag_top_df = sag_top_df.sort_values(['round_MCC'], ascending=[False])
    sag_top_df['rank'] = (sag_top_df['round_MCC']).astype(float).rank(method='dense', ascending=False).astype(float)
    rank_df = sag_top_df[['sag_id', 'nu', 'gamma', 'rank', 'precision', 'MCC', 'sensitivity']]
    top_MCC = sag_top_df['round_MCC'].iloc[0]
    sag_sub_df = sag_top_df.loc[sag_top_df['round_MCC'] >= top_MCC]
    best_sub_list.append(sag_sub_df)
    rank_sub_list.append(rank_df)

concat_df = pd.concat(best_sub_list)
#concat_df = concat_df.loc[concat_df['gamma'] != 'scale']
#concat_df['gamma'] = pd.to_numeric(concat_df['gamma'], errors='coerce')
# select the config that overfits the least
concat_df['gamma_sorter'] = [gamma_dict[x] for x in concat_df['gamma']]
concat_df = concat_df.sort_values(['round_MCC', 'nu', 'gamma_sorter'],
                                   ascending=[False, False, True])
sag_dedup_df = concat_df.drop_duplicates(subset='sag_id', keep='first')

keep_list = ['sag_id', 'level', 'inclusion', 'transformation', 'gamma', 'nu', 'precision','MCC', 'sensitivity']
pred_stack_df = sag_dedup_df[keep_list].set_index(['sag_id', 'level', 'inclusion',
                                              'transformation', 'gamma', 'nu']).stack().reset_index()
pred_stack_df.columns = ['sag_id', 'level', 'inclusion', 'transformation',
                         'gamma', 'nu', 'metric', 'score'
                         ]
pred_stack_df['nu_gamma'] = [str(x[0]) + '_' + str(x[1]) for x in 
                             zip(pred_stack_df['nu'], pred_stack_df['gamma'])
                             ]
pred_stack_df['gamma_sorter'] = [gamma_dict[x] for x in pred_stack_df['gamma']]
pred_stack_df = pred_stack_df.sort_values(['nu', 'gamma_sorter'], ascending=[True, True])

ax = sns.catplot(x="metric", y="score", hue="gamma", kind='box',
                    data=pred_stack_df, aspect=2, palette=sns.light_palette("black"),
                    flierprops=flierprops)
plt.savefig('PR_plots/gamma_boxplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

ax = sns.catplot(x="metric", y="score", hue="nu", kind='box',
                    data=pred_stack_df, aspect=2, palette=sns.light_palette("black"),
                    flierprops=flierprops)
plt.savefig('PR_plots/nu_boxplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

count_df = sag_dedup_df[['nu', 'gamma', 'nu_gamma']].groupby(['nu', 'gamma']).count().reset_index()
count_df.columns = ['nu', 'gamma', 'count']
count_df = count_df.sort_values(['count'], ascending=[False])
top_nu = count_df['nu'].iloc[0]
top_gamma = count_df['gamma'].iloc[0]

nu_order = list(sorted(set(count_df['nu'])))
g = sns.FacetGrid(count_df, col="gamma", col_wrap=4)
g.map(sns.barplot, "nu", "count", order=nu_order, ci=None)
plt.savefig('PR_plots/nu_gamma_facetplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

# Rank the configs per sag
rank_concat_df = pd.concat(rank_sub_list)
#rank_concat_df = rank_concat_df.loc[rank_concat_df['gamma'] != 'scale']
#rank_concat_df['gamma'] = pd.to_numeric(rank_concat_df['gamma'], errors='coerce')
best_config_df = rank_concat_df.loc[((rank_concat_df['nu'] == top_nu) &
                                     (rank_concat_df['gamma'] == top_gamma)
                                     )]
mean_P = best_config_df['precision'].mean()*100
mean_MCC = best_config_df['MCC'].mean()*100
mean_R = best_config_df['sensitivity'].mean()*100
max_P = best_config_df['precision'].max()*100
max_MCC = best_config_df['MCC'].max()*100
max_R = best_config_df['sensitivity'].max()*100
min_P = best_config_df['precision'].min()*100
min_MCC = best_config_df['MCC'].min()*100
min_R = best_config_df['sensitivity'].min()*100
#tot_count = len(best_config_df['sag_id'])
#top_5 = (len(best_config_df['rank'].loc[best_config_df['rank'] <= 5])/tot_count)*100
#top_10 = (len(best_config_df['rank'].loc[best_config_df['rank'] <= 10])/tot_count)*100
#top_20 = (len(best_config_df['rank'].loc[best_config_df['rank'] <= 20])/tot_count)*100
g = sns.displot(best_config_df, x="rank")
text_str = ''.join(['Mean:\n  P=', str(mean_P.round(2)), '\n  R=', str(mean_R.round(2)),
                    '\n  MCC=', str(mean_MCC.round(2)),
                    #'\nMax:\n  P=', str(max_P.round(2)), '\n  R=', str(max_R.round(2)),
                    #'\n  MCC=', str(max_MCC.round(2)),
                    #'\nMin:\n  P=', str(min_P.round(2)), '\n  R=', str(min_R.round(2)),
                    #'\n  MCC=', str(min_MCC.round(2))
                    ])
for ax in g.axes.flat:
  ax.text(30, 800, text_str, fontsize=9)
plt.savefig('PR_plots/best_config_histo.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()
print(top_level, top_inclusion, top_transformation, top_nu, top_gamma)

