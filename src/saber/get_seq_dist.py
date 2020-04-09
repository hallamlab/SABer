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

from sklearn import linear_model
from minepy import MINE

from collections import Counter

from scipy.linalg import eigh, cholesky

from statsmodels.sandbox.stats.runs import runstest_2samp


def roundup(x, step):
    return int(math.ceil(x / float(step))) * step


# Create models from data
def best_fit_distribution(data, bins, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [(d,getattr(st, d))  for d in _distn_names if 'levy_stable' not in d]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for dname, distribution in DISTRIBUTIONS:
        # Try to fit the distribution
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
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params, best_sse)


def test_dist_fit(data, orig_data):

    stat_results = st.wilcoxon(orig_data, data) #

    return stat_results


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
            while ((int(sub_v) < min_int) | (int(sub_v) > max_int)):
                sub_v = dist.rvs(loc=loc, scale=scale, *arg)
            v.append(sub_v)
    else:
        v = dist.rvs(loc=loc, scale=scale, size=size, *arg)
    return v


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
    fasta_dir2 = sys.argv[3]
    fasta_list2 = [os.path.join(fasta_dir2, x) for x in os.listdir(fasta_dir2) if (('.fasta' in x)
                  | ('.fna' in x))
                  ]

    fasta_list = fasta_list1 + fasta_list2

    sag_seq_list = []
    for i, fasta_file in enumerate(fasta_list):
        print(fasta_file)
        sag_id = i
        fasta_dat = s_utils.get_seqs(fasta_file)
        orig_cont_cnt = len(fasta_dat)
        contig_cnt = len([x[1] for x in fasta_dat if len(x[1]) >= contig_len_min])
        seq_rec_list = []
        j = 0
        for z, seq_rec in enumerate(fasta_dat):
            header, seq = seq_rec
            if len(seq) >= contig_len_min: # removes seqs shorter than 1000 bp
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

# Print basic metrics for real SAGs
max_contigs = max(list(sag_seq_df['contig_cnt']))
min_contigs = min(list(sag_seq_df['contig_cnt']))
sag_count = len(list(sag_seq_df['sag_id'].unique()))
max_contig_len = max(list(sag_seq_df['seq_len']))
min_contig_len = min(list(sag_seq_df['seq_len']))
print('Basic metrics for Real SAGs')
print('SAG count: %s' % sag_count)
print('Max contigs: %s' % max_contigs)
print('Min contigs: %s' % min_contigs)
print('Max contig length: %s' % max_contig_len)
print('Min contig length: %s' % min_contig_len)

# Build SAG contig count df
#sag_cnt_interval = pd.interval_range(start=0, freq=1, end=roundup(max_contigs, 100))
sag_contig_df = df = sag_seq_df.drop_duplicates(subset='sag_id', keep="first")

# Build interval for number of bins using step and max contig len across all SAGs
#interval_range = pd.interval_range(start=0, freq=step, end=max_len)
#sag_seq_df['n-tiles'] = pd.cut(sag_seq_df['seq_len'], bins=interval_range).astype(str)

'''
cnt_bins_df = sag_seq_df.groupby(['sag_id','n-tiles'])['contig_id'].count().reset_index()
cnt_bins_df.columns = ['sag_id', 'n-tiles', 'count']
cnt_bins_df['bins'] = [int(x.split(',')[0].strip('(')) for x in cnt_bins_df['n-tiles']]
cnt_bins_df.sort_values(by=['bins'], inplace=True)
#print(cnt_bins_df.head())
piv_df = cnt_bins_df.pivot(index='bins', columns='sag_id', values='count')
piv_df.fillna(0, inplace=True)
piv_df.reset_index(inplace=True)
#print(piv_df.head())
piv_df.to_csv('SAG_Catelog_bins.tsv', sep='\t', index=False)
'''

# Calc covariance, pearson, spearman, and MIC for contig count vs. contig len
X = sag_seq_df['contig_cnt'].values
y = sag_seq_df['seq_len'].values
SAG_covariance = np.cov(X, y)
pearsons, _ = st.pearsonr(X, y)
spearmans, _ = st.spearmanr(X, y)
print(SAG_covariance)
print('Pearsons correlation: %.4f' % pearsons)
print('Spearmans correlation: %.4f' % spearmans)
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



'''
# Build various plots for SAG raw data
sag_seq_df = pd.read_csv('SAG_Catelog_contig_length.tsv', sep='\t', header=0)
print(sag_seq_df.head())

for x in list(sag_seq_df['best_dist'].unique()):
    sub_ss_df = sag_seq_df.loc[sag_seq_df['best_dist'] == x]
    print(x, len(list(sub_ss_df['sag_id'].unique())))

sss_df = sag_seq_df[['sag_id', 'contig_cnt', 'best_dist']].drop_duplicates()

print(sss_df.head())

g = sns.FacetGrid(sag_seq_df, col="best_dist", col_wrap=8, margin_titles=True, sharey=False,
                    sharex=False
                    )
g.map(plt.scatter, "contig_cnt",  "seq_len")
g.savefig('scatter_facet.pdf')
plt.clf()
'''
g = sns.scatterplot(x='contig_cnt', y='seq_len', data=sag_seq_df)
g.figure.savefig('scatter.pdf')
plt.clf()
'''
g = sns.kdeplot(data=sag_seq_df['contig_cnt'])
g.figure.savefig('kde_contig_cnt.pdf')
plt.clf()

g = sns.kdeplot(data=sag_seq_df['seq_len'])
g.figure.savefig('kde_contig_len.pdf')
plt.clf()

sys.exit()
'''




# Find best dist for real data contig count
data2 = sag_contig_df['contig_cnt']
# Find best fit distribution
best_fit_name, best_fit_params, best_sse = best_fit_distribution(data2, max_contigs)
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

# Building syn_N number of synthetic distributions and then select the best fitting one.
syn_N = 10000
best_syn_stats = (1000, 0) # Dummy values for stats
best_syn_contigs = []
for N in range(syn_N):
    synth_contig_count = [int(abs(x)) for x in
                          gen_random(best_dist, best_fit_params, len(data2), min_contigs,
                                        max_contigs
                                        )
                          ]
    comp_stats = test_dist_fit(synth_contig_count, list(data2))
    if comp_stats[1] >= best_syn_stats[1]:
        best_syn_stats = comp_stats
        best_syn_contigs = synth_contig_count

print('Wilcoxon signed-rank test for synthetic data: statistic=%s, p-value=%s'
       % best_syn_stats
       )
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
stat_str = 'Wilcoxon signed-rank test for synthetic data: statistic=%s, p-value=%s' % best_syn_stats
ax.set_title(u'Synthetic SAG Contig Count with best fit distribution \n' + dist_str +
                u'\n' + stat_str)
ax.set_xlabel(u'Contig Count')
ax.set_ylabel('Frequency')
plt.savefig('Synthetic_SAGs_Dist_Contig_Count.pdf')
plt.clf()

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
'''

# Find best dist for real SAGs contig seq len, sep SAGs by contig count (step 10)
count_step = 20 # magic number for step below
sag_count_list = np.arange(1, roundup(max_contigs, 100), count_step)
sag_count_dist_dict = {}
for s_cnt in sag_count_list:
    print(s_cnt)
    sag_sub_df = sag_seq_df.loc[(sag_seq_df['contig_cnt'] >= s_cnt) &
                                (sag_seq_df['contig_cnt'] < s_cnt + count_step)
                                ]
    if sag_sub_df.shape[0] != 0:
        max_sagcont_len = max(list(sag_sub_df['seq_len']))
        sag_len_interval = pd.interval_range(start=0, freq=step, end=roundup(max_sagcont_len, step))
        data = sag_sub_df['seq_len']
        # Find best fit distribution
        best_fit_name, best_fit_params, best_sse = best_fit_distribution(data, len(sag_len_interval))
        best_dist = getattr(st, best_fit_name)
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
        ax.set_title(u'SAG Catelog Contig Length with best fit distribution \n' + dist_str)
        ax.set_xlabel(u'Contig Length (bp)')
        ax.set_ylabel('Frequency')
        plt.savefig('Real_SAGs_Dist_Contig_Length_' + str(s_cnt) + '.pdf')
        plt.clf()
        plt.close()
        sag_count_dist_dict[s_cnt] = (best_fit_name, best_fit_params, best_sse, best_dist)

# Building syn_N number of synthetic distributions and then select the best fitting one.
# complicated way of binning the synth SAG counts even when there are missing bins :/
synth_sag_map = {}
for c in best_syn_contigs:
    for s in list(sag_count_dist_dict.keys()).sort():
        if c >= s:
            if c < s + count_step:
                synth_sag_map[c] = s
            else:
                synth_sag_map[c] = last_s
        last_s = s

syn_N = 1000
best_len_stats = (None, 0)
best_sag_list = []
for N in range(syn_N):
    # get lengths for all synth contigs
    syn_sag_list = []
    syn_contig_list = []
    for i, synth_sag in enumerate(best_syn_contigs):
        s_key = synth_sag_map[synth_sag]
        best_dist = sag_count_dist_dict[s_key]
        best_fit_params = sag_count_dist_dict[s_key]
        cont_len_list = [int(abs(x)) for x in
                         gen_random(best_dist, best_fit_params, synth_sag, contig_len_min,
                                    max_contig_len
                                    )
                         ]
        for s_contig, cont_len in enumerate(cont_len_list):
            syn_sag_list.append([i, s_contig, cont_len, synth_sag])
            syn_contig_list.append(cont_len)

    comp_stats = test_dist_fit(syn_contig_list, list(data))
    if comp_stats[1] >= best_len_stats[1]:
        best_len_stats = comp_stats
        best_sag_list = syn_sag_list
print('Wilcoxon signed-rank test for synthetic data: statistic=%s, p-value=%s'
       % best_len_stats
       )

syn_sag_df = pd.DataFrame(best_sag_list, columns=['sag_id', 'contig_id', 'seq_len', 'contig_cnt'])
syn_sag_df.to_csv('Synth-SAG_contig_length.tsv', sep='\t', index=False)

# Print basic metrics for synthetic SAGs
max_contigs = max(list(syn_sag_df['contig_cnt']))
min_contigs = min(list(syn_sag_df['contig_cnt']))
sag_count = len(list(syn_sag_df['sag_id'].unique()))
max_contig_len = max(list(syn_sag_df['seq_len']))
min_contig_len = min(list(syn_sag_df['seq_len']))
print('Basic metrics for Synthetic SAGs')
print('SAG count: %s' % sag_count)
print('Max contigs: %s' % max_contigs)
print('Min contigs: %s' % min_contigs)
print('Max contig length: %s' % max_contig_len)
print('Min contig length: %s' % min_contig_len)


# Calc covariance, pearson, spearman, and MIC for contig count vs. contig len
X = syn_sag_df['contig_cnt'].values
y = syn_sag_df['seq_len'].values
covariance = np.cov(X, y)
pearsons, _ = st.pearsonr(X, y)
spearmans, _ = st.spearmanr(X, y)
print(covariance)
print('Pearsons correlation: %.3f' % pearsons)
print('Spearmans correlation: %.3f' % spearmans)
sys.exit()
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

