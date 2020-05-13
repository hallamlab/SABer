import saber.utilities as s_utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import math

import warnings
import numpy as np
import scipy.stats as st
import statsmodels.api as sm
import matplotlib
from scipy.stats._continuous_distns import _distn_names
from scipy.stats._continuous_distns import _distn_names
#from scipy.stats._discrete_distns import _distn_names as _d_distn_names

from sklearn import linear_model
#from minepy import MINE

from collections import Counter

from scipy.linalg import eigh, cholesky

from statsmodels.sandbox.stats.runs import runstest_2samp

import logging

import multiprocessing

import random

import statistics as stat

from scipy import interpolate

pd.set_option('float_format', '{:f}'.format)
pd.options.mode.chained_assignment = None


def roundup(x, step):
    return int(math.ceil(x / float(step))) * step


def build_dist(p):
    # Estimate distribution parameters from data
    # Try to fit the distribution
    dname, distribution, data, x, y = p
    try:
        # Ignore warnings from data that can't be fit
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            # fit dist to data
            params = distribution.fit(data)

            # Separate parts of parameters
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # Calculate fitted PDF and error with fit in distribution
            pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
            isf = distribution.isf([0.1, 0.05, 0.01], loc=loc, scale=scale, *arg)
            sse = np.sum(np.power(y - pdf, 2.0))
    except:
        params = (0.0, 1.0)
        isf = []
        sse = np.inf

    return (dname, distribution, params, sse, isf)


# Create models from data
def best_fit_distribution(data, bins, num_cores=multiprocessing.cpu_count()):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [(d,getattr(st, d), data, x, y) for d in _distn_names
                        if 'levy_stable' not in d
                        ]
    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    best_isf = []

    #print("Building multiprocessing pool")
    pool = multiprocessing.Pool(processes=num_cores)
    results = pool.imap_unordered(build_dist, DISTRIBUTIONS)
    #print("Executing pool")
    merge_list = []
    for i, o in enumerate(results):
        sys.stderr.write('\r   done {0:.0%}'.format(i/len(DISTRIBUTIONS)))
        merge_list.append(o)
    pool.close()
    pool.join()

    for dname, distribution, params, sse, isf in merge_list:
        # identify if this distribution is better
        if best_sse > sse > 0:
            best_distribution = distribution
            best_params = params
            best_sse = sse
            best_isf = isf

    return (best_distribution.name, best_params, best_sse, best_isf)


def test_dist_fit(data, orig_data, method):
    if method == 'ad':
        stat_results = st.anderson_ksamp([orig_data, data])
    elif method == 'ks':
        stat_results = st.ks_2samp(orig_data, data) #

    return stat_results


def ks_crit_val(n_trials, alpha):
    return st.ksone.ppf(1-alpha/2, n_trials)


def make_pdf(dist, params, size=200):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale
                     ) if arg else dist.ppf(0.01, loc=loc, scale=scale
                     )
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale,
                    ) if arg else dist.ppf(0.99, loc=loc, scale=scale
                    )

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


def build_syn_count(syn_N, best_dist, best_fit_params, orig_data, min_contigs, max_contigs,
                    num_cores=multiprocessing.cpu_count()
                    ):

    best_syn_stats = False # Dummy values for stats
    best_syn_contigs = []

    syn_param_list = [(best_dist, best_fit_params, orig_data, min_contigs, max_contigs) for x in
                        range(syn_N)
                        ]

    print("Building multiprocessing pool")
    pool = multiprocessing.Pool(processes=num_cores)
    results = pool.imap_unordered(syn_count_dist, syn_param_list)
    print("Executing pool")
    merge_list = []
    for i, o in enumerate(results):
        sys.stderr.write('\rdone {0:.0%}'.format(i/len(syn_param_list)))
        merge_list.append(o)
    pool.close()
    pool.join()

    for comp_stats, synth_contig_count in merge_list:
        if best_syn_stats == False:
            best_syn_stats = comp_stats
            best_syn_contigs = synth_contig_count

        elif ((comp_stats.significance_level > best_syn_stats.significance_level) &
            (comp_stats.statistic < best_syn_stats.statistic)
            ):
            best_syn_stats = comp_stats
            best_syn_contigs = synth_contig_count

    return (best_syn_stats, best_syn_contigs)


def syn_count_dist(p):
    best_dist, best_fit_params, orig_data, min_contigs, max_contigs = p
    synth_contig_count = [int(abs(x)) for x in
                          gen_random(best_dist, best_fit_params, len(orig_data), min_contigs,
                                        max_contigs
                                        )
                          ]
    comp_stats = test_dist_fit(synth_contig_count, list(orig_data))

    return (comp_stats, synth_contig_count)


def gen_random(dist, params, size, min_int=0, max_int=0):
    """Generate distributions's Random Number
    returns an array of len size containing random floats from dist
    """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    if ((min_int != 0) & (max_int != 0)):
        v = []
        for x in range(size):
            sub_v = dist.rvs(loc=loc, scale=scale, *arg)
            while ((sub_v < min_int) | (sub_v > max_int)):
                sub_v = dist.rvs(loc=loc, scale=scale, *arg)
            v.append(sub_v)
    else:
        v = dist.rvs(loc=loc, scale=scale, size=size, *arg)
    return v


logging.basicConfig(filename='build_synthetic_dataset.log',level=logging.DEBUG)

# magic values
step = 1000
contig_len_min = 2000
######################

 # open all SAGs FASTAs and count contigs and lengths
if sys.argv[1] == 'True':
    fasta_dir1 = sys.argv[2]
    fasta_list1 = [os.path.join(fasta_dir1, x) for x in os.listdir(fasta_dir1) if (('.fasta' in x)
                  | ('.fna' in x))
                  ]
    '''
    fasta_dir2 = sys.argv[3]
    fasta_list2 = [os.path.join(fasta_dir2, x) for x in os.listdir(fasta_dir2) if (('.fasta' in x)
                  | ('.fna' in x))
                  ]

    fasta_list = fasta_list1 + fasta_list2
    '''
    fasta_list = fasta_list1

    sag_seq_list = []
    for i, fasta_file in enumerate(fasta_list):
        print(fasta_file)
        logging.info(fasta_file)
        sag_id = i
        fasta_dat = s_utils.get_seqs(fasta_file)
        orig_cont_cnt = len(fasta_dat)
        contig_cnt = len([x[1] for x in fasta_dat if len(x[1]) >= contig_len_min])
        seq_rec_list = []
        j = 0
        for z, seq_rec in enumerate(fasta_dat):
            header, seq = seq_rec
            if len(seq) >= contig_len_min: # removes seqs shorter than N bp
                seq_rec_list.append([sag_id, j, len(seq), orig_cont_cnt, contig_cnt])
                j += 1
        sag_seq_list.extend(seq_rec_list)

    sag_seq_df = pd.DataFrame(sag_seq_list, columns=['sag_id', 'contig_id', 'seq_len', 'orig_cont_cnt',
                                                    'contig_cnt']
                                                    )
    sag_seq_df.to_csv('Real-SAG_contig_length.tsv', sep='\t', index=False)

else:
    sag_seq_df = pd.read_csv('Real-SAG_contig_length.tsv', sep='\t', header=0)

max_len = roundup(max(list(sag_seq_df['seq_len'])), step)

# trim out outlier high contig count SAGs
max_contigs = max(list(sag_seq_df['contig_cnt']))
trim_cnt_list = []
contig_temp_list = range(1, max_contigs+1)
for m in contig_temp_list:
    if m in list(sag_seq_df['contig_cnt']):
        trim_cnt_list.append(True)
    else:
        trim_cnt_list.append(False)
false_cnt = []
for i, v in enumerate(trim_cnt_list):
    if v == True:
        false_cnt = []
    else:
        false_cnt.append(v)
    if len(false_cnt) >= 10:
        max_contigs = contig_temp_list[i-10]
        break
sag_seq_df = sag_seq_df.loc[sag_seq_df['contig_cnt'] <= max_contigs]

# Print basic metrics for real SAGs
min_contigs = min(list(sag_seq_df['contig_cnt']))
sag_count = len(list(sag_seq_df['sag_id'].unique()))
max_contig_len = max(list(sag_seq_df['seq_len']))
min_contig_len = min(list(sag_seq_df['seq_len']))
print('Basic metrics for Real SAGs')
logging.info('Basic metrics for Real SAGs')
print('SAG count: %s' % sag_count)
logging.info('SAG count: %s' % sag_count)
print('Max contigs: %s' % max_contigs)
logging.info('Max contigs: %s' % max_contigs)
print('Min contigs: %s' % min_contigs)
logging.info('Min contigs: %s' % min_contigs)
print('Max contig length: %s' % max_contig_len)
logging.info('Max contig length: %s' % max_contig_len)
print('Min contig length: %s' % min_contig_len)
logging.info('Min contig length: %s' % min_contig_len)

# Build SAG contig count df
sort_sag_seq_df = sag_seq_df.sort_values(by=['seq_len'], ascending=False)
sag_contig_df = sort_sag_seq_df.drop_duplicates(subset='sag_id', keep="first")

# Build min/max contig len for sags
sag_min_max_dict = {}
for r in range(1, max_contigs + 1):
    sag_cnt_df = sag_seq_df.loc[sag_seq_df['contig_cnt'] == r]
    if sag_cnt_df.shape[0] != 0:
        cnt_max = sag_cnt_df['seq_len'].max()
        cnt_min = sag_cnt_df['seq_len'].min()
        sag_min_max_dict[r] = (cnt_min, cnt_max)
    else:
        sag_min_max_dict[r] = sag_min_max_dict[r-1]

# Calc covariance, pearson, spearman, and MIC for contig count vs. contig len
X = sag_seq_df['contig_cnt'].values
y = sag_seq_df['seq_len'].values
SAG_covariance = np.cov(X, y)
pearsons, _ = st.pearsonr(X, y)
spearmans, _ = st.spearmanr(X, y)
print(SAG_covariance)
logging.info(SAG_covariance)
print('Pearsons correlation: %.4f' % pearsons)
logging.info('Pearsons correlation: %.4f' % pearsons)
print('Spearmans correlation: %.4f' % spearmans)
logging.info('Spearmans correlation: %.4f' % spearmans)

'''
def print_stats(mine):
    print('MIC: %.4f' % mine.mic())
    print('MAS: %.4f' % mine.mas())
    print('MEV: %.4f' % mine.mev())
    print('MCN (eps=0): %.4f' % mine.mcn(0))
    print('MCN (eps=1-MIC): %.4f' % mine.mcn_general())
    print('GMIC: %.4f' % mine.gmic())
    print('TIC: %.4f' % mine.tic())

mine = MINE(alpha=0.6, c=15, est="mic_approx")
mine.compute_score(X, y)
print_stats(mine)
'''

# Build various plots for SAG raw data
g = sns.scatterplot(x='contig_cnt', y='seq_len', data=sag_seq_df)
plt.savefig('Real_SAGs_scatter.pdf')
plt.clf()
plt.close()
'''
# Find best dist for real data contig count
data2 = sag_contig_df['contig_cnt']
# Find best fit distribution
print("Finding best distribution for real contig count data")
best_fit_name, best_fit_params, best_sse, best_isf = best_fit_distribution(data2, max_contigs)
best_dist = getattr(st, best_fit_name)
# Make PDF with best params
pdf = make_pdf(best_dist, best_fit_params, size=len(data2))
# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data2.plot(kind='hist', bins=max_contigs, density=True, alpha=0.5,
           label='Data', legend=True, ax=ax
           )
param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)
ax.set_title(u'SAG Catelog Contig Count with best fit distribution \n' + dist_str)
ax.set_xlabel(u'Contig Count')
ax.set_ylabel('Frequency')
plt.savefig('Real_SAGs_Dist_Contig_Count.pdf')
plt.clf()
'''

# Create a custom distribution using the real data
data2 = sag_contig_df['contig_cnt']
cont_occ_df = sag_contig_df['contig_cnt'].value_counts().rename_axis('unique_values').reset_index(name='counts')
cont_occ_dict = {x[0]:x[1] for x in zip(cont_occ_df['unique_values'], cont_occ_df['counts'])}
cont_occ_df['Hz'] = [x/cont_occ_df['counts'].sum() for x in cont_occ_df['counts']]
print(cont_occ_df.head())
cont_occ_df.sort_values(by='unique_values', inplace=True)
#xk = cont_occ_df['unique_values']
#pk = cont_occ_df['Hz']

# build interpolated contig counts
f = interpolate.interp1d(cont_occ_df['unique_values'], cont_occ_df['counts'])
xk = np.arange(1, max_contigs+1, 1)
y_new = [int(round(y)) for y in f(xk)]
for i, y in enumerate(y_new):
    if y == 1:
        r_int = random.randint(0, 101)
        if r_int < 50:
            y_new[i] = 0
pk = [y/sum(y_new) for y in y_new]
inter_contig_cnt = []
for x, y in zip(xk, y_new):
    for y1 in range(y):
        inter_contig_cnt.append(x)

real_cnt_dist = st.rv_discrete(name='cnt_dist', values=(xk, pk))

#fig, ax = plt.subplots(1, 1)
#ax.plot(xk, custm.pmf(xk), 'ro', ms=12, mec='r')
#ax.vlines(xk, 0, custm.pmf(xk), colors='r', lw=4)

pmf = pd.Series(real_cnt_dist.pmf(xk), xk)
# Display
plt.figure(figsize=(12,8))
#ax = pmf.plot(lw=2, label='PMF', legend=True)
ax = sns.kdeplot(inter_contig_cnt, color='k')
data2.plot(kind='hist', bins=max_contigs, density=True, alpha=1.0,
           label='Data', legend=False, ax=ax, color='orange'
           )
#param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
#param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
#dist_str = '{}({})'.format(best_fit_name, param_str)
#ax.set_title(u'SAG Catelog Contig Count with best fit distribution \n' + dist_str)
ax.set_xlabel(u'Contig Count')
ax.set_ylabel('Frequency')
plt.savefig('Real_SAGs_Dist_Contig_Count.pdf')
plt.clf()
plt.close()
'''
# Building syn_N number of synthetic distributions and then select the best fitting one.
print("\nFinding synthetic contig count data from real data distribution")
syn_N = 10000
best_syn_stats, best_syn_contigs = build_syn_count(syn_N, best_dist, best_fit_params, data2,
                                                    min_contigs, max_contigs
                                                    )
'''
'''
alphas = [0.1, 0.05, 0.01]
# Print table headers
print('\n\n{:<6}|{:<6} Level of significance, alpha'.format(' ', ' '))
print('{:<6}|{:>8} {:>8} {:>8} {:>8}'.format(*['Trials'] + alphas))
print('-' * 42)
# Print critical values for each n_trials x alpha combination
print('{:6d}|{:>8.5f} {:>8.5f} {:>8.5f} {:>8.5f}'.format(
                                    *[len(best_syn_contigs)] + [f for f in best_isf]
                                    )
        )
'''
syn_N = 1000
best_syn_stats = False
best_syn_contigs = []
for N in range(syn_N):
    sys.stderr.write('\r Count {0:.0%}'.format(N/syn_N))
    syn_contig_cnt = pd.Series([int(x) for x in real_cnt_dist.rvs(size=len(inter_contig_cnt))])
    comp_stats = test_dist_fit(syn_contig_cnt, data2, 'ad')
    if best_syn_stats == False:
        best_syn_stats = comp_stats
        best_syn_contigs = syn_contig_cnt
    elif best_syn_stats.statistic > comp_stats.statistic: # best_len_stats.significance_level < comp_stats.significance_level: # ((
        best_syn_stats = comp_stats
        best_syn_contigs = syn_contig_cnt
#best_syn_contigs = pd.Series([int(x) for x in real_cnt_dist.rvs(size=len(data2))])
#best_syn_stats = test_dist_fit(data2, best_syn_contigs, 'ad')

crit_vals = [str(round(c, 4)) for c in best_syn_stats.critical_values]
print('\n\nAnderson-Darling 2 sample test for synthetic data: statistic=%s, p-value=%s, critical_values=%s'
        % (round(best_syn_stats.statistic, 4), round(best_syn_stats.significance_level, 2), ', '.join(crit_vals))
        )
logging.info('\n\nAnderson-Darling 2 sample test for synthetic data: statistic=%s, p-value=%s, critical_values=%s'
        % (round(best_syn_stats.statistic, 4), round(best_syn_stats.significance_level, 2), ', '.join(crit_vals))
        )

# Plot Real SAG best dist fitted to synthetic data
syn_contig_df = pd.DataFrame(best_syn_contigs, columns=['syn_contig_cnt'])
data = syn_contig_df['syn_contig_cnt']
# Display
plt.figure(figsize=(12,8))
ax = sns.kdeplot(inter_contig_cnt, color='k')
syn_contig_df.plot(kind='hist', bins=max(list(syn_contig_df['syn_contig_cnt'])),
                    density=True, alpha=0.5,
                    label='Data', legend=True, ax=ax, color='b'
                    )
#param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
#param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
#dist_str = '{}({})'.format(best_fit_name, param_str)
stat_str = 'Anderson-Darling 2 sample test for synthetic data: statistic=%s, p-value=%s' \
        % (round(best_syn_stats.statistic, 4), round(best_syn_stats.significance_level, 2))
ax.set_title(u'Synthetic SAG Contig Count with best fit distribution \n' + stat_str)
ax.set_xlabel(u'Contig Count')
ax.set_ylabel('Frequency')
plt.savefig('Synthetic_SAGs_Dist_Contig_Count.pdf')
plt.clf()
plt.close()
'''
# Find best dist for real SAGs contig seq len, all combined
max_sagcont_len = max(list(sag_seq_df['seq_len']))
sag_len_interval = pd.interval_range(start=0, freq=step, end=roundup(max_sagcont_len, step))
data = sag_seq_df['seq_len']
# Find best fit distribution
best_fit_name, best_fit_params, best_sse = best_fit_distribution(data, len(sag_len_interval))
best_dist = getattr(st, best_fit_name)
# Make PDF with best params
pdf = make_pdf(best_dist, best_fit_params)
# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=len(sag_len_interval), density=True, alpha=0.5,
          label='Data', legend=True, ax=ax
          )
param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)
ax.set_title(u'SAG Catelog Contig Length with best fit distribution \n' + dist_str)
ax.set_xlabel(u'Contig Length (bp)')
ax.set_ylabel('Frequency')
plt.savefig('Real_SAGs_Dist_Contig_Length' + '.pdf')
plt.clf()
plt.close()

# Building syn_N number of synthetic distributions and then select the best fitting one.
syn_N = 1000
best_len_stats = (None, 0)
best_sag_list = []
for N in range(syn_N):
    # get lengths for all synth contigs
    syn_sag_list = []
    syn_contig_list = []
    for i, synth_sag in enumerate(best_syn_contigs):
        cont_len_list = [int(abs(x)) for x in
                         gen_random(best_dist, best_fit_params, synth_sag, contig_len_min,
                                    max_contig_len
                                    )
                         ]
        for s_contig, cont_len in enumerate(cont_len_list):
            syn_sag_list.append([i, s_contig, cont_len, synth_sag])
            syn_contig_list.append(cont_len)
    comp_stats = test_dist_fit(syn_contig_list, list(sag_seq_df['seq_len']))
    if comp_stats[1] >= best_len_stats[1]:
        best_len_stats = comp_stats
        best_sag_list = syn_sag_list
    print(best_len_stats, comp_stats)
print('Mann-Whitney rank test for synthetic data: statistic=%s, p-value=%s'
       % best_len_stats
       )

syn_sag_df = pd.DataFrame(best_sag_list, columns=['sag_id', 'contig_id', 'seq_len', 'contig_cnt'])
syn_sag_df.to_csv('Synth-SAG_contig_length.tsv', sep='\t', index=False)

# Build scatter plot for synthetic data
g = sns.scatterplot(x='contig_cnt', y='seq_len', data=syn_sag_df)
g.figure.savefig('Synthetic_SAGs_scatter.pdf')
plt.clf()

# Plot Real SAG best dist fitted to synthetic data
syn_contig_df = pd.DataFrame(best_syn_contigs, columns=['syn_contig_cnt'])
data = syn_contig_df['syn_contig_cnt']
# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=max(data), density=True, alpha=0.5,
          label='Data', legend=True, ax=ax
          )
param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)
stat_str = 'Mann-Whitney rank test for synthetic data: statistic=%s, p-value=%s' % best_syn_stats
ax.set_title(u'Synthetic SAG Contig Count with best fit distribution \n' + dist_str +
                u'\n' + stat_str)
ax.set_xlabel(u'Contig Count')
ax.set_ylabel('Frequency')
plt.savefig('Synthetic_SAGs_Dist_Contig_Count.pdf')
plt.clf()
'''
# Find best dist for real SAGs contig seq len, sep SAGs by contig count
sag_len_dist_dict = {}
sag_count_dist_dict = {}
for each in range(1, max_contigs + 1):
    sys.stderr.write('\r Real-len {}'.format(each))
    sag_sub_df = sag_seq_df.loc[sag_seq_df['contig_cnt'] == each]
    if sag_sub_df.shape[0] != 0:
        max_sagcont_len = max(list(sag_sub_df['seq_len']))
        sag_len_interval = pd.interval_range(start=min_contig_len-1, freq=step, end=roundup(max_sagcont_len+step, step))
        sag_sub_df['intervals'] = pd.cut(sag_sub_df['seq_len'], bins=sag_len_interval)
        sag_sub_df['left'] = [x.left for x in sag_sub_df['intervals']]
        # Create a custom distribution using the real data
        len_data = sag_sub_df['left']
        cont_len_df = sag_sub_df['left'].value_counts().rename_axis('unique_values').reset_index(name='counts')
        cont_len_dict = {x[0]:x[1] for x in zip(cont_len_df['unique_values'], cont_len_df['counts'])}
        cont_len_df['Hz'] = [x/cont_len_df['counts'].sum() for x in cont_len_df['counts']]
        cont_len_df.sort_values(by='unique_values', inplace=True)
        xk = cont_len_df['unique_values']
        pk = cont_len_df['Hz']
        real_len_dist = st.rv_discrete(name='len_dist', values=(xk, pk))
        pmf = pd.Series(real_len_dist.pmf(xk), xk)
        # Display
        plt.figure(figsize=(12,8))
        ax = sns.kdeplot(len_data, color='k')
        len_data.plot(kind='hist', bins=len(sag_len_interval), density=True, alpha=1.0,
                   label='Data', legend=True, ax=ax, color='orange'
                   )
        ax.set_xlabel(u'Contig Length')
        ax.set_ylabel('Frequency')
        plt.savefig('real_length_plots/Real_SAGs_Dist_Contig_Length_' + str(each) + '.pdf')
        plt.clf()
        plt.close()

        sag_len_dist_dict[each] = real_len_dist
        '''
        data = sag_sub_df['seq_len']
        best_fit_name, best_fit_params, best_sse, best_isf = best_fit_distribution(data, len(sag_len_interval))
        best_dist = getattr(st, best_fit_name)
        sag_count_dist_dict[each] = (best_fit_name, best_fit_params, best_sse, best_dist)
        '''
    else:
        sag_len_dist_dict[each] = sag_len_dist_dict[each - 1]
        #sag_count_dist_dict[each] = sag_count_dist_dict[each - 1]

# Building syn_N number of synthetic distributions and then select the best fitting one.
syn_N = 100
#best_len_stats = False
best_sag_dict = {x[0]:(False, []) for x in zip(range(len(best_syn_contigs)), best_syn_contigs)}
real_len_dat = list(sag_seq_df['seq_len'])
real_len_dat.sort(reverse=False)
break_list = [x for x in range(len(best_syn_contigs))]*10
print('\n')
for N in range(syn_N):
    # get lengths for all synth contigs
    syn_sag_list = []
    syn_contig_list = []
    #for i, synth_sag in enumerate(best_syn_contigs):
    for i in break_list:
        synth_sag = best_syn_contigs[i]
        sys.stderr.write('\r                             ')
        sys.stderr.write('\r Synth-len {}:{}:{}'.format(N, i, synth_sag))
        '''
        best_dist = sag_count_dist_dict[synth_sag][3]
        best_fit_params = sag_count_dist_dict[synth_sag][1]
        cont_len_list = [int(abs(x)) for x in
                         gen_random(best_dist, best_fit_params, synth_sag, contig_len_min,
                                    sag_min_max_dict[synth_sag][1]
                                    )
                         ]
        '''
        best_dist = sag_len_dist_dict[synth_sag]
        cont_inter_list = list(best_dist.rvs(size=synth_sag))
        cont_len_list = [random.randint(x, x + step) for x in cont_inter_list]
        best_len_stats, best_sag_list = best_sag_dict[i]
        sag_i_df = sag_seq_df.loc[sag_seq_df['contig_cnt'] == synth_sag]
        sag_i_len = list(sag_i_df['seq_len'])
        if sag_i_len == []:
            ssag = synth_sag
            while sag_i_len == []:
                ssag = ssag - 1
                sag_i_df = sag_seq_df.loc[sag_seq_df['contig_cnt'] == ssag]
                sag_i_len = list(sag_i_df['seq_len'])
        comp_stats = test_dist_fit(cont_len_list, sag_i_len, 'ad')
        if best_len_stats == False:
            best_sag_dict[i] = (comp_stats, cont_len_list)
        elif ((0.05 <= comp_stats.significance_level) & # best_len_stats.significance_level
                (sum(cont_len_list) > sum(best_sag_dict[i][1]) &
                (comp_stats.statistic < comp_stats.critical_values[2]))
                ): # best_len_stats.statistic > comp_stats.statistic
            best_sag_dict[i] = (comp_stats, cont_len_list)
            break_list.remove(i)
    '''
        for s_contig, cont_len in enumerate(cont_len_list):
            syn_sag_list.append([i, s_contig, cont_len, synth_sag])
            syn_contig_list.append(cont_len)
    syn_contig_list.sort(reverse=False)
    comp_stats = test_dist_fit(syn_contig_list, real_len_dat, 'ad')
    print(len(syn_contig_list))
    print(len(real_len_dat))
    print(syn_contig_list[:10])
    print(real_len_dat[:10])
    print(comp_stats)
    print(sum(syn_contig_list)/len(syn_contig_list))
    print(stat.median(syn_contig_list))
    if best_len_stats == False:
        best_len_stats = comp_stats
        best_sag_list = syn_sag_list
    elif best_len_stats.statistic > comp_stats.statistic: # best_len_stats.significance_level < comp_stats.significance_level: # ((
        best_len_stats = comp_stats
        best_sag_list = syn_sag_list
    #if best_len_stats.significance_level >= 0.25:
    #   break
    print(best_len_stats)
    '''

syn_sag_list = []
for k in best_sag_dict.keys():
    i_sag_stats, i_sag_len = best_sag_dict[k]
    for s, i_cont in enumerate(i_sag_len):
        r_list = [k, s, i_cont, len(i_sag_len), i_sag_stats.statistic,
                    i_sag_stats.significance_level,
                    ','.join([str(x) for x in i_sag_stats.critical_values])
                    ]
        syn_sag_list.append(r_list)
'''
crit_vals = [str(round(c, 4)) for c in best_len_stats.critical_values]
print('\n\nAnderson-Darling 2 sample test for synthetic data: statistic=%s, p-value=%s, critical_values=%s'
        % (round(best_len_stats.statistic, 4), round(best_len_stats.significance_level, 2), ', '.join(crit_vals))
        )
logging.info('\n\nAnderson-Darling 2 sample test for synthetic data: statistic=%s, p-value=%s, critical_values=%s'
        % (round(best_len_stats.statistic, 4), round(best_len_stats.significance_level, 2), ', '.join(crit_vals))
        )
'''
syn_sag_df = pd.DataFrame(syn_sag_list, columns=['sag_id', 'contig_id', 'seq_len', 'contig_cnt',
                            'statistic', 'p_value', 'crit_values']
                            )
syn_sag_df.to_csv('Synth-SAG_contig_length.tsv', sep='\t', index=False)

# Print basic metrics for synthetic SAGs
max_contigs = max(list(syn_sag_df['contig_cnt']))
min_contigs = min(list(syn_sag_df['contig_cnt']))
sag_count = len(list(syn_sag_df['sag_id'].unique()))
max_contig_len = max(list(syn_sag_df['seq_len']))
min_contig_len = min(list(syn_sag_df['seq_len']))
print('\nBasic metrics for Synthetic SAGs')
logging.info('Basic metrics for Synthetic SAGs')
print('SAG count: %s' % sag_count)
logging.info('SAG count: %s' % sag_count)
print('Max contigs: %s' % max_contigs)
logging.info('Max contigs: %s' % max_contigs)
print('Min contigs: %s' % min_contigs)
logging.info('Min contigs: %s' % min_contigs)
print('Max contig length: %s' % max_contig_len)
logging.info('Max contig length: %s' % max_contig_len)
print('Min contig length: %s' % min_contig_len)
logging.info('Min contig length: %s' % min_contig_len)


# Calc covariance, pearson, spearman, and MIC for contig count vs. contig len
X = syn_sag_df['contig_cnt'].values
y = syn_sag_df['seq_len'].values
covariance = np.cov(X, y)
pearsons, _ = st.pearsonr(X, y)
spearmans, _ = st.spearmanr(X, y)
print(covariance)
logging.info(covariance)
print('Pearsons correlation: %.3f' % pearsons)
logging.info('Pearsons correlation: %.3f' % pearsons)
print('Spearmans correlation: %.3f' % spearmans)
logging.info('Spearmans correlation: %.3f' % spearmans)

'''
def print_stats(mine):
    print("MIC", mine.mic())
    print("MAS", mine.mas())
    print("MEV", mine.mev())
    print("MCN (eps=0)", mine.mcn(0))
    print("MCN (eps=1-MIC)", mine.mcn_general())
    print("GMIC", mine.gmic())
    print("TIC", mine.tic())

mine = MINE(alpha=0.6, c=15, est="mic_approx")
mine.compute_score(X, y)
print_stats(mine)
'''

# Build various plots for SAG raw data
g = sns.scatterplot(x='contig_cnt', y='seq_len', data=syn_sag_df)
plt.savefig('Synthetic_SAGs_scatter.pdf')
plt.clf()
plt.close()


sys.exit()

















# Plot Real SAG best dist fitted to synthetic data
data = syn_sag_df['seq_len']
# Make PDF with best params
pdf = make_pdf(best_dist, best_fit_params, size=len(data))
# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=len(sag_len_interval), density=True, alpha=0.5,
          label='Data', legend=True, ax=ax
          )
param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)
ax.set_title(u'Synthetic SAG Contig Length with best fit distribution \n' + dist_str)
ax.set_xlabel(u'Contig Length (bp)')
ax.set_ylabel('Frequency')
plt.savefig('Synthetic_SAGs_Dist_Contig_Length.pdf')
plt.clf()
plt.close()

# Reproduce correlation b/t Contig count and length
# Choice of cholesky or eigenvector method.
method = 'cholesky'
#method = 'eigenvectors'

# We need a matrix `c` for which `c*c^T = r`.  We can use, for example,
# the Cholesky decomposition, or the we can construct `c` from the
# eigenvectors and eigenvalues.

if method == 'cholesky':
    # Compute the Cholesky decomposition.
    c = cholesky(SAG_covariance, lower=True)
else:
    # Compute the eigenvalues and eigenvectors.
    evals, evecs = eigh(SAG_covariance)
    # Construct c, so c*c^T = r.
    c = np.dot(evecs, np.diag(np.sqrt(evals)))

dat_array = np.array([X, y])
# Convert the data to correlated random variables.
corr_dat = np.dot(c, dat_array)
corr_df = pd.DataFrame(np.transpose(corr_dat), columns=['contig_cnt', 'seq_len'])
corr_df['sag_id'] = syn_sag_df['sag_id']
corr_df['contig_id'] = syn_sag_df['contig_id']
corr_df.to_csv('Synth-SAG_contig_length_correlated.tsv', sep='\t', index=False)

# Calc covariance, pearson, spearman, and MIC for contig count vs. contig len
X = corr_df['contig_cnt'].values
y = corr_df['seq_len'].values

covariance = np.cov(X, y)
pearsons, _ = st.pearsonr(X, y)
spearmans, _ = st.spearmanr(X, y)
print(covariance)
print('Pearsons correlation: %.3f' % pearsons)
print('Spearmans correlation: %.3f' % spearmans)
'''
def print_stats(mine):
    print("MIC", mine.mic())
    print("MAS", mine.mas())
    print("MEV", mine.mev())
    print("MCN (eps=0)", mine.mcn(0))
    print("MCN (eps=1-MIC)", mine.mcn_general())
    print("GMIC", mine.gmic())
    print("TIC", mine.tic())

mine = MINE(alpha=0.6, c=15, est="mic_approx")
mine.compute_score(X, y)
print_stats(mine)
'''

