import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns;

sns.set(style="ticks", color_codes=True)
import sys
import os
import math

pd.set_option('display.max_columns', None)

'''
# Trup Positive Venn Diagram
err_path = './'
tp_file = err_path + 'TruePos_table.tsv'
truePos_df = pd.read_csv(tp_file, header=0, sep='\t')
venn_dict = {'100': 0, '010': 0, '110': 0, '001': 0, '101': 0, '011': 0, '111': 0}
for sag_id in set(truePos_df['sag_id']):
    sag_tp_df = truePos_df.loc[truePos_df['sag_id'] == sag_id]
    for contig_id in set(sag_tp_df['@@SEQUENCEID']):
        contig_sub_df = sag_tp_df.loc[sag_tp_df['@@SEQUENCEID'] == contig_id]
        algo_list = list(contig_sub_df['algo'])
        # Count venn categories
        if (('MinHash' in algo_list) & ('TPM' not in algo_list) &
            ('tetra_comb' not in algo_list)
            ):
            venn_dict['100'] = venn_dict['100'] + 1
        elif (('MinHash' not in algo_list) & ('TPM' in algo_list) &
            ('tetra_comb' not in algo_list)
            ):
            venn_dict['010'] = venn_dict['010'] + 1
        elif (('MinHash' not in algo_list) & ('TPM' not in algo_list) &
            ('tetra_comb' in algo_list)
            ):
            venn_dict['001'] = venn_dict['001'] + 1
        elif (('MinHash' in algo_list) & ('TPM' in algo_list) &
            ('tetra_comb' not in algo_list)
            ):
            venn_dict['110'] = venn_dict['110'] + 1
        elif (('MinHash' in algo_list) & ('TPM' not in algo_list) &
            ('tetra_comb' in algo_list)
            ):
            venn_dict['101'] = venn_dict['101'] + 1
        elif (('MinHash' not in algo_list) & ('TPM' in algo_list) &
            ('tetra_comb' in algo_list)
            ):
            venn_dict['011'] = venn_dict['011'] + 1
        elif (('MinHash' in algo_list) & ('TPM' in algo_list) &
            ('tetra_comb' in algo_list)
            ):
            venn_dict['111'] = venn_dict['111'] + 1
v3 = venn3(subsets = {'100':1, '010':1, '110':1,
                      '001':1, '101':1, '011':1, '111':1},
           set_labels = ('MinHash', 'Abundance', 'Tetranucleotide'))

v3.get_patch_by_id('100').set_color('red')
v3.get_patch_by_id('010').set_color('yellow')
v3.get_patch_by_id('001').set_color('blue')
v3.get_patch_by_id('110').set_color('orange')
v3.get_patch_by_id('101').set_color('purple')
v3.get_patch_by_id('011').set_color('green')
v3.get_patch_by_id('111').set_color('grey')

v3.get_label_by_id('100').set_text(venn_dict['100'])
v3.get_label_by_id('010').set_text(venn_dict['010'])
v3.get_label_by_id('001').set_text(venn_dict['001'])
v3.get_label_by_id('110').set_text(venn_dict['110'])
v3.get_label_by_id('101').set_text(venn_dict['101'])
v3.get_label_by_id('011').set_text(venn_dict['011'])
v3.get_label_by_id('111').set_text(venn_dict['111'])

for text in v3.subset_labels:
    text.set_fontsize(8)
plt.savefig(err_path + "SABer_strain_venn.png", bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

print('Venn built')
'''

sns.set_context("poster")
sns.set_style('whitegrid')
sns.set(font_scale=0.75)
err_path = sys.argv[1]

algo_path = err_path + 'multi-algo'
if not os.path.exists(algo_path):
    os.makedirs(algo_path)
level_path = err_path + 'multi-level'
if not os.path.exists(level_path):
    os.makedirs(level_path)
comp_path = err_path + '/Comp_plots/'
if not os.path.exists(comp_path):
    os.makedirs(comp_path)

sns_colors = list(sns.color_palette("muted"))

err_file = err_path + '/All_stats_count.tsv'
err_df = pd.read_csv(err_file, header=0, sep='\t')
map_algo = {'synSAG': 'synSAG', 'minhash': 'MinHash', 'mbn_abund': 'MBN-Abund', 'tetra_gmm': 'GMM',
            'tetra_svm': 'OCSVM', 'tetra_iso': 'Isolation Forest', 'tetra_comb': 'Tetra Ensemble',
            'xpg': 'SABer-xPG'
            }
err_df['algorithm'] = [map_algo[x] for x in err_df['algorithm']]
err_df['level'] = ['exact' if x == 'perfect' else x for x in err_df['level']]
err_trim_df = err_df.loc[err_df['statistic'] != 'F1_score']

sns.set(font_scale=1.5)  # crazy big
level_list = ['strain']
stat_list = ['sensitivity', 'precision']
trim_df = err_trim_df.loc[((err_trim_df['level'].isin(level_list)) &
                           (err_trim_df['statistic'].isin(stat_list))
                           )]
trim_df['sag_id'] = [x.replace('.synSAG', '') for x in trim_df['sag_id']]
sagid_list = list(trim_df['sag_id'].loc[trim_df['algorithm'] == 'SABer-xPG'])
trim_df = trim_df.loc[trim_df['sag_id'].isin(sagid_list)]


def myceil(x, base=5):
    return base * math.ceil(x / base)


interval_labels = {5: '[0,5]', 10: '(5,10]', 15: '(10,15]', 20: '(15,20]', 25: '(20,25]', 30: '(25,30]',
                   35: '(30,35]', 40: '[35,40]', 45: '(40,45]', 50: '(45,50]', 55: '(50,55]', 60: '(55,60]',
                   65: '(60,65]', 70: '(65,70]', 75: '[70,75]', 80: '(75,80]', 85: '(80,85]', 90: '(85,90]',
                   95: '(90,95]', 100: '(95,100]'
                   }
trim_df['percent'] = [x * 100 for x in trim_df['score']]
trim_df['round_percent'] = [myceil(x) for x in trim_df['percent']]
trim_df = trim_df[['sag_id', 'algorithm', 'statistic', 'score', 'percent', 'round_percent']]
sag_df = trim_df[['sag_id', 'percent', 'round_percent']
].loc[((trim_df['algorithm'] == 'synSAG') & (trim_df['statistic'] == 'sensitivity'))]
merge_df = trim_df.merge(sag_df, on=['sag_id'])
merge_df.columns = ['sag_id', 'stage', 'statistic', 'score', 'stage_score', 'stage_round_score',
                    'synSAG_score', 'synSAG_score_cat'
                    ]
merge_df['syn_SAG_label'] = [interval_labels[x] for x in merge_df['synSAG_score_cat']]
filter_df = merge_df.loc[merge_df['stage'] != 'synSAG']
sensitivity_df = filter_df.loc[filter_df['statistic'] == 'sensitivity']
precision_df = filter_df.loc[filter_df['statistic'] == 'precision']
synSAG_df = merge_df.loc[((merge_df['stage'] == 'synSAG') &
                          (merge_df['statistic'] == 'sensitivity')
                          )]

sensitivity_df['datatype'] = 'Sensitivity'
precision_df['datatype'] = 'Precision'
synSAG_df['datatype'] = 'synSAG_Sensitivity'
syn_stage_sense_df = pd.concat([sensitivity_df, synSAG_df])

df_list = []
df_list.extend([sensitivity_df, precision_df])
for algo in set(filter_df['stage']):
    tmp_df = synSAG_df.copy()
    tmp_df['stage'] = algo
    df_list.append(tmp_df)
concat_df = pd.concat(df_list)
g = sns.catplot(x='synSAG_score_cat', y='stage_score', hue='datatype', col='stage',
                kind='box', col_wrap=3,
                col_order=['MinHash', 'MBN-Abund', 'Isolation Forest', 'OCSVM', 'GMM',
                           'SABer-xPG'
                           ],
                palette={'synSAG_Sensitivity': sns_colors[2], 'Sensitivity': sns_colors[1], 'Precision': sns_colors[0]},
                data=concat_df
                )
[plt.setp(ax.texts, text="") for ax in g.axes.flat]
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.set_titles(row_template='{row_name}', col_template='{col_name}')

g.savefig(err_path + '/SABer_Sensitivity_boxplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

sys.exit()

sns.set_context("poster")
sns.set_style('whitegrid')
sns.set(font_scale=0.75)
stage_list = ['MinHash', 'SABer-xPG']
sens_trim_df = sensitivity_df.loc[sensitivity_df['stage'].isin(stage_list)]
g = sns.relplot(x='stage', y='stage_score', hue='synSAG_score_cat', style='synSAG_score_cat',
                kind='line', ci=95, data=sens_trim_df, palette='muted'
                )
plt.ylim(0, 100)
[plt.setp(ax.texts, text="") for ax in g.axes.flat]
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.set_titles(row_template='{row_name}', col_template='{col_name}')

g.savefig(err_path + '/SABer_MinHash_relplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

for algo in set(syn_stage_sense_df['stage']):
    # Plot Before and after SAG -> SABer-xPG completness
    algo_list = ['synSAG', algo]
    sub_trim_df = syn_stage_sense_df.loc[syn_stage_sense_df['stage'].isin(algo_list)]
    sns.set_context("poster")
    sns.set_style('whitegrid')
    sns.set(font_scale=0.75)
    g = sns.JointGrid(data=sub_trim_df, x='synSAG_score', y='stage_score', hue='stage',
                      ylim=(0, 101)
                      )
    g.plot_joint(sns.scatterplot)
    g.plot_marginals(sns.histplot, kde=True)

    g.savefig(comp_path + algo + "_synSAG_Completeness.png", bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()

    sns.set_context("poster")
    sns.set_style('whitegrid')
    sns.set(font_scale=0.75)
    g = sns.catplot(x='synSAG_score_cat', y='stage_score', hue='stage', kind='box',
                    data=sub_trim_df, linewidth=0.5
                    )
    g.savefig(comp_path + algo + "_synSAG_Completeness_box.png", bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()

unstack_df = err_trim_df.set_index(['sag_id', 'algorithm', 'level', 'statistic']).unstack('statistic')
unstack_df.reset_index(inplace=True)
unstack_df.columns = ['sag_id', 'algorithm', 'level', 'MCC', 'Precision',
                      'Sensitivity'
                      ]
val_df_list = []
outlier_list = []
for algo in set(unstack_df['algorithm']):
    algo_df = unstack_df.loc[unstack_df['algorithm'] == algo]
    for level in set(unstack_df['level']):
        level_df = algo_df.loc[algo_df['level'] == level].set_index(
            ['sag_id', 'algorithm', 'level']
        )
        for stat in ['MCC', 'Precision', 'Sensitivity']:
            stat_df = level_df[[stat]]
            mean = list(stat_df.mean())[0]
            var = list(stat_df.var())[0]
            skew = list(stat_df.skew())[0]
            kurt = list(stat_df.kurt())[0]
            IQ_25 = list(stat_df.quantile(0.25))[0]
            IQ_75 = list(stat_df.quantile(0.75))[0]
            IQ_10 = list(stat_df.quantile(0.10))[0]
            IQ_90 = list(stat_df.quantile(0.90))[0]
            IQ_05 = list(stat_df.quantile(0.05))[0]
            IQ_95 = list(stat_df.quantile(0.95))[0]
            IQ_01 = list(stat_df.quantile(0.01))[0]
            IQ_99 = list(stat_df.quantile(0.99))[0]
            IQR = IQ_75 - IQ_25
            # calc Tukey Fences
            upper_bound = IQ_75 + (1.5 * IQR)
            lower_bound = IQ_25 - (1.5 * IQR)
            header_list = ['algorithm', 'level', 'stat', 'mean', 'var', 'skew', 'kurt',
                           'IQ_25', 'IQ_75', 'IQ_10', 'IQ_90', 'IQ_05', 'IQ_95',
                           'IQ_01', 'IQ_99', 'IQR (25-75)', 'upper_bound', 'lower_bound'
                           ]
            val_list = [algo, level, stat, mean, var, skew, kurt, IQ_25, IQ_75,
                        IQ_10, IQ_90, IQ_05, IQ_95, IQ_01, IQ_99, IQR, upper_bound,
                        lower_bound
                        ]
            val_df = pd.DataFrame([val_list], columns=header_list)
            val_df_list.append(val_df)
            stat_df['statistic'] = stat
            stat_df.reset_index(inplace=True)
            stat_df.columns = ['sag_id', 'algorithm', 'level', 'score', 'statistic']
            stat_df = stat_df[['sag_id', 'algorithm', 'level', 'statistic', 'score']]
            outlier_df = stat_df.loc[(stat_df['score'] < lower_bound) &
                                     (stat_df['score'] < 0.99)]
            outlier_list.append(outlier_df)

concat_val_df = pd.concat(val_df_list)
concat_val_df.to_csv(err_path + '/Compiled_stats.tsv', sep='\t', index=False)
concat_out_df = pd.concat(outlier_list)
concat_out_df.to_csv(err_path + '/Compiled_outliers.tsv', sep='\t', index=False)
level_order = ['domain', 'family', 'class', 'order', 'genus', 'species', 'strain', 'exact']

flierprops = dict(markerfacecolor='0.75', markersize=5, markeredgecolor='w',
                  linestyle='none')

for level in set(err_trim_df['level']):
    level_df = err_trim_df.loc[err_trim_df['level'] == level]
    sns.set_context("paper")
    ax = sns.catplot(x="statistic", y="score", hue='algorithm', kind='box',
                     data=level_df, aspect=2, palette=sns.light_palette("black"),
                     flierprops=flierprops)
    plt.plot([-0.5, 3.5], [0.25, 0.25], linestyle='--', alpha=0.3, color='k')
    plt.plot([-0.5, 3.5], [0.50, 0.50], linestyle='--', alpha=0.3, color='k')
    plt.plot([-0.5, 3.5], [0.75, 0.75], linestyle='--', alpha=0.3, color='k')

    plt.ylim(0, 1)
    plt.xlim(-0.5, 3.5)
    # plt.title('SAG-plus CAMI-1-High error analysis')
    ax._legend.set_title('Workflow\nStage')
    plt.savefig(err_path + '/multi-algo/' + level + '_error_boxplox_count.png',
                bbox_inches='tight'
                )
    plt.clf()
    plt.close()

# build multi-level precision boxplot
stat_list = ['precision', 'sensitivity', 'MCC']
for algo in set(err_trim_df['algorithm']):
    comb_stat_df = err_trim_df.loc[((err_trim_df['algorithm'] == algo) &
                                    (err_trim_df['level'].isin(level_order)) &
                                    (err_trim_df['statistic'].isin(stat_list))
                                    )]
    # concat_stat_df = pd.concat([mock_stat_df, comb_stat_df])
    sns.set_context("paper")
    ax = sns.catplot(x="level", y="score", hue='statistic', kind='box',
                     data=comb_stat_df, aspect=2, palette=sns.light_palette("black"),
                     flierprops=flierprops)

    plt.plot([-0.5, 4.5], [0.25, 0.25], linestyle='--', alpha=0.3, color='k')
    plt.plot([-0.5, 4.5], [0.50, 0.50], linestyle='--', alpha=0.3, color='k')
    plt.plot([-0.5, 4.5], [0.75, 0.75], linestyle='--', alpha=0.3, color='k')

    # plt.ylim(0, 1)
    # plt.xlim(-0.5, 4.5)
    plt.title('SABer ' + algo + ' by Taxonomic-level')

    plt.savefig(err_path + '/multi-level/' + algo.replace(' ', '_') + '_multi-level_boxplox_count.png',
                bbox_inches='tight'
                )
    plt.clf()
    plt.close()

# Stat by level line plot
err_deduped_df = err_trim_df.loc[err_trim_df['algorithm'].isin(['GMM', 'OCSVM',
                                                                'Isolation Forest',
                                                                'Tetra Ensemble', 'SABer-xPG'])
]

sns.set(font_scale=1.5)  # crazy big
g = sns.relplot(x='level', y='score', hue='statistic', style='statistic',
                col='algorithm', kind='line', col_wrap=5,
                col_order=['Isolation Forest', 'OCSVM', 'GMM', 'Tetra Ensemble', 'SABer-xPG'],
                sort=False,
                data=err_deduped_df
                )
[plt.setp(ax.texts, text="") for ax in g.axes.flat]
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.set_titles(row_template='{row_name}', col_template='{col_name}')

g.savefig(err_path + '/SABer_relplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

g = sns.relplot(x='level', y='score', hue='statistic', style='statistic',
                col='algorithm', kind='line', col_wrap=3, ci=95,
                col_order=['MinHash', 'MBN-Abund', 'Isolation Forest', 'OCSVM', 'GMM',
                           'SABer-xPG'],
                sort=False,
                data=err_trim_df
                )
plt.ylim(0, 1)
# plt.xlim(0, 1)
[plt.setp(ax.texts, text="") for ax in g.axes.flat]
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.set_titles(row_template='{row_name}', col_template='{col_name}')

g.savefig(err_path + '/SABer_AllSteps_relplot.png', bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

# Open SAG ID to AMBER map
s2a_map_file = err_path + '/SABer2AMBER_map.tsv'
s2a_map_df = pd.read_csv(s2a_map_file, header=0, sep='\t')
s2a_map_df['sag_id'] = [x.rsplit('.', 1)[0] for x in s2a_map_df['sag_id']]
deduped_df = unstack_df.loc[unstack_df['algorithm'].isin(['Isolation Forest', 'OCSVM', 'GMM', 'SABer-xPG'])]

deduped_df['synthSAG_id'] = deduped_df['sag_id']
deduped_df['sag_id'] = [x.rsplit('.', 2)[0] for x in deduped_df['sag_id']]
deduped_amb_df = pd.merge(deduped_df, s2a_map_df, on='sag_id', how='left')
deduped_amb_df.columns = ['bin_id', 'algorithm', 'level', 'MCC', 'Precision',
                          'Sensitivity', 'synthSAG_id', 'genome_id'
                          ]
strain_df = deduped_amb_df.loc[deduped_amb_df['level'] == 'strain']
strain_filter_df = strain_df[['bin_id', 'algorithm', 'Precision', 'Sensitivity', 'genome_id']]

# AMBER Genome
amber_gen_file = err_path + '/AMBER_genome_results.tsv'
amber_gen_df = pd.read_csv(amber_gen_file, header=0, sep='\t')
amber_gen_df.columns = ['sample_id', 'bin_id', 'genome_id', 'Precision', 'Sensitivity',
                        'Predicted size (bp)', 'True positives (bp)', 'True size (bp)',
                        'Purity (seq)', 'Completeness (seq)', 'Predicted size (seq)',
                        'True positives (seq)', 'True size (seq)', 'algorithm'
                        ]
amber_filter_df = amber_gen_df[['bin_id', 'algorithm', 'Precision', 'Sensitivity', 'genome_id']]

sab_amb_df = pd.concat([strain_filter_df, amber_filter_df])

avg_sab_amb_df = sab_amb_df.groupby(['algorithm']).mean().reset_index()
filter_sab_amb_df = avg_sab_amb_df.loc[~avg_sab_amb_df['algorithm'].isin(['Isolation Forest', 'OCSVM', 'GMM'])]
paired_cmap_dict = {'blue': (0.2823529411764706, 0.47058823529411764, 0.8156862745098039),
                    'orange': (0.9333333333333333, 0.5215686274509804, 0.2901960784313726),
                    'green': (0.41568627450980394, 0.8, 0.39215686274509803),
                    'rose': (0.8392156862745098, 0.37254901960784315, 0.37254901960784315),
                    'purple': (0.5843137254901961, 0.4235294117647059, 0.7058823529411765),
                    'brown': (0.5490196078431373, 0.3803921568627451, 0.23529411764705882),
                    'pink': (0.8627450980392157, 0.49411764705882355, 0.7529411764705882),
                    'gray': (0.4745098039215686, 0.4745098039215686, 0.4745098039215686),
                    'gold': (0.8352941176470589, 0.7333333333333333, 0.403921568627451),
                    'lightblue': (0.5098039215686274, 0.7764705882352941, 0.8862745098039215)
                    }
# Plot average stats for algorithm
color_dict = {'Binsanity-wf_0.2.5.9': paired_cmap_dict['rose'],
              'Binsanity_0.2.5.9': paired_cmap_dict['rose'],
              'COCACOLA': paired_cmap_dict['purple'],
              'CONCOCT_2': paired_cmap_dict['lightblue'],
              'CONCOCT_CAMI': paired_cmap_dict['lightblue'],
              'DAS_Tool_1.1': paired_cmap_dict['blue'],
              'MaxBin_2.0.2_CAMI': paired_cmap_dict['gold'],
              'MaxBin_2.2.4': paired_cmap_dict['gold'],
              'MetaBAT_2.11.2': paired_cmap_dict['green'],
              'MetaBAT_CAMI': paired_cmap_dict['green'],
              'Metawatt_3.5_CAMI': paired_cmap_dict['brown'],
              'MyCC_CAMI': paired_cmap_dict['gray'],
              'SABer-xPG': paired_cmap_dict['orange']
              }
marker_dict = {'MaxBin_2.0.2_CAMI': 'o', 'MetaBAT_CAMI': 'o',
               'MaxBin_2.2.4': 'd', 'CONCOCT_2': 'o',
               'Binsanity_0.2.5.9': 'o', 'Binsanity-wf_0.2.5.9': 'd',
               'SABer-xPG': 's', 'COCACOLA': 'o', 'Metawatt_3.5_CAMI': 'o',
               'CONCOCT_CAMI': 'd',
               'MyCC_CAMI': 'o', 'DAS_Tool_1.1': 'd',
               'MetaBAT_2.11.2': 'd'
               }
cat_order = ['Binsanity-wf_0.2.5.9', 'Binsanity_0.2.5.9', 'COCACOLA', 'CONCOCT_2',
             'CONCOCT_CAMI', 'DAS_Tool_1.1', 'MaxBin_2.0.2_CAMI', 'MaxBin_2.2.4',
             'MetaBAT_2.11.2', 'MetaBAT_CAMI', 'Metawatt_3.5_CAMI', 'MyCC_CAMI',
             'SABer-xPG'
             ]
g = sns.scatterplot(y='Precision', x='Sensitivity', hue='algorithm', palette=color_dict,
                    style='algorithm', markers=marker_dict, hue_order=cat_order,
                    data=filter_sab_amb_df, s=75
                    )

g.set(ylabel='Average Precision', xlabel='Average Sensitivity')
plt.xlim(0, 1)
plt.ylim(0, 1.05)

plt.legend(edgecolor='b', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
g.figure.savefig(err_path + '/AMBER_SABer_scatterplot.png', bbox_inches='tight', dpi=300)

flierprops = dict(markerfacecolor='0.75', markersize=5, markeredgecolor='w',
                  linestyle='none')

piv_sab_amb_df = sab_amb_df.set_index(['bin_id', 'algorithm', 'genome_id']).stack().reset_index()
piv_sab_amb_df.columns = ['bin_id', 'algorithm', 'genome_id', 'statistic', 'score']

cat_order = ['Binsanity-wf_0.2.5.9', 'Binsanity_0.2.5.9', 'COCACOLA', 'CONCOCT_2',
             'CONCOCT_CAMI', 'DAS_Tool_1.1', 'MaxBin_2.0.2_CAMI', 'MaxBin_2.2.4',
             'MetaBAT_2.11.2', 'MetaBAT_CAMI', 'Metawatt_3.5_CAMI', 'MyCC_CAMI',
             'Isolation Forest', 'OCSVM', 'GMM', 'SABer-xPG'
             ]
with sns.axes_style("white"):
    ax = sns.catplot(x="statistic", y="score", hue='algorithm', kind='box',
                     data=piv_sab_amb_df, aspect=2, palette=sns.light_palette("black"),
                     flierprops=flierprops, hue_order=cat_order)
    plt.plot([-1, 3], [0.25, 0.25], linestyle='--', alpha=0.3, color='k')
    plt.plot([-1, 3], [0.50, 0.50], linestyle='--', alpha=0.3, color='k')
    plt.plot([-1, 3], [0.75, 0.75], linestyle='--', alpha=0.3, color='k')

    plt.ylim(0, 1)
    plt.xlim(-0.5, 1.5)
    # plt.title('SAG-plus CAMI-1-High error analysis')
    ax._legend.set_title('Workflow\nStage')
    plt.savefig(err_path + '/AMBER_SABer_error_boxplox_count.png', bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()
