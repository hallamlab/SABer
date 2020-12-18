from song.song import SONG
import sys
import pandas as pd
from sklearn import decomposition
import numpy as np
from skbio.stats.composition import clr
import pyfastx
from collections import Counter
from itertools import product, islice



def get_kmer(seq, n):
        "Returns a sliding window (of width n) over data from the iterable"
        "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                "
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result


'''
fasta_file = '/home/rmclaughlin/sharknado/Sandbox/Ryan/SABer/SABer_stdout_3000/subcontigs/CAMI_high_GoldStandardAssembly.3000.subcontigs.fasta'

fasta = pyfastx.Fasta(fasta_file)

# Dict of all tetramers
tetra_cnt_dict = {''.join(x):[] for x in product('atgc', repeat=4)}
header_list = []
# count up all tetramers and also populate the tetra dict
for rec in fasta:
    header = rec.name
    header_list.append(header)
    seq = rec.seq
    tmp_dict = {k: 0 for k, v in tetra_cnt_dict.items()}
    clean_seq = seq.strip('\n').lower()
    comp_dict = {'a': 't', 't': 'a', 'g': 'c', 'c': 'g'}
    comp_seq = ''.join([comp_dict[x] if x in comp_dict else x for x in clean_seq])
    kmer_list = [''.join(x) for x in get_kmer(clean_seq, 4)]
    comp_kmer_list = [''.join(x) for x in get_kmer(comp_seq, 4)]
    tetra_counter = Counter(kmer_list + comp_kmer_list)
    total_kmer_cnt = sum(tetra_counter.values())
    # add counter to tmp_dict
    for tetra in tmp_dict.keys():
        count_tetra = int(tetra_counter[tetra])
        tmp_dict[tetra] = count_tetra
    # map tetras to their reverse tetras (not compliment)
    dedup_dict = {}
    for tetra in tmp_dict.keys():
        if (tetra not in dedup_dict.keys()) & (tetra[::-1]
            not in dedup_dict.keys()
            ):
            dedup_dict[tetra] = ''
        elif tetra[::-1] in dedup_dict.keys():
            dedup_dict[tetra[::-1]] = tetra
    # combine the tetras and their reverse (not compliment)
    tetra_prop_dict = {}
    for tetra in dedup_dict.keys():
        if dedup_dict[tetra] != '':
            tetra_prop_dict[tetra] = tmp_dict[tetra] + tmp_dict[dedup_dict[tetra]]
            #t_prop = (tmp_dict[tetra]
            #            + tmp_dict[dedup_dict[tetra]]) / total_kmer_cnt
            #tetra_prop_dict[tetra] = t_prop
        else:
            tetra_prop_dict[tetra] = tmp_dict[tetra]
            #t_prop = tmp_dict[tetra] / total_kmer_cnt
            #tetra_prop_dict[tetra] = t_prop
    # add to tetra_cnt_dict
    for k in tetra_cnt_dict.keys():
        if k in tetra_prop_dict.keys():
            tetra_cnt_dict[k].append(tetra_prop_dict[k])
        else:
            tetra_cnt_dict[k].append(0.0)
# convert the final dict into a pd dataframe for ease
tetra_cnt_dict['contig_id'] = header_list
tetra_cnt_df = pd.DataFrame.from_dict(tetra_cnt_dict).set_index('contig_id')
print(tetra_cnt_df.head())

dedupped_df = tetra_cnt_df.loc[:, (tetra_cnt_df != 0.0).any(axis=0)]
dedupped_df += 1 # add pseudocount
print(dedupped_df.head())

first_val = dedupped_df.columns[0]
last_val = dedupped_df.columns[-1]
dedupped_df['sum'] = dedupped_df.sum(axis=1)
normal_df = dedupped_df.loc[:, first_val:last_val].div(dedupped_df['sum'], axis=0)
print(normal_df.head())

clr_df = normal_df.apply(clr)
#clr_df['contig_id'] = mg_headers
#clr_df.set_index('contig_id', inplace=True)
print(clr_df.head())
clr_df.reset_index().to_csv('clr_trans/CAMI_high_GoldStandardAssembly.CLR.compliment.tetras.tsv', sep='\t',
                            index=False
                            )
'''
clr_df = pd.read_csv('clr_trans/CAMI_high_GoldStandardAssembly.CLR.compliment.tetras.tsv', sep='\t',
                     header=0, index_col='contig_id'
                     )
for n in [10, 20, 40, 80]:
    #print('Transforming data with SONG, using ' + str(n) + ' components')
    #song_model = SONG(n_components=n) #, min_dist=0, n_neighbors=1)
    #song_trans = song_model.fit_transform(clr_df.values)
    #song_df = pd.DataFrame(song_trans, index=clr_df.index.values)
    #song_df.reset_index().to_csv('clr_trans/CAMI_high_GoldStandardAssembly.CLR.SONG.' + str(n) + '.tsv',
    #                             sep='\t', index=False
    #                             )
    
    print('Transforming data with PCA, using ' + str(n) + ' components')
    pca = decomposition.PCA(n_components=n)
    pca.fit(clr_df.values)
    pca_trans = pca.transform(clr_df.values)
    print('Explained variance = ', sum(pca.explained_variance_ratio_))
    pca_df = pd.DataFrame(pca_trans, index=clr_df.index.values)
    pca_df.reset_index().to_csv('clr_trans/CAMI_high_GoldStandardAssembly.CLR.compliment.PCA.' + str(n) + '.tsv',
                                sep='\t', index=False
                                )