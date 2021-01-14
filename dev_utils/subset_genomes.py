import os
import random
import sys

import pandas as pd


def build_subseq(seq_rec, length):
    s_len = int(seq_rec['seq_len'])
    d_len = s_len - length
    r_int = random.randint(0, d_len)
    seq_str = seq_rec['seq'].values[0]
    sub_seq = seq_str[r_int:r_int + length]
    f_seq, r_seq = seq_str.split(sub_seq)
    contig_id = seq_rec['contig_id'].values[0]
    s = '|'
    if s in contig_id:
        trim_contig_id = contig_id.rsplit(s, 1)[0]
        coords = contig_id.rsplit(s, 1)[1].split('_')
        start = coords[0]
        corr_start = int(start) + int(r_int)
        corr_end = corr_start + int(length)
        f_id = trim_contig_id + s + str(start) + '_' + str(corr_start)
        r_id = trim_contig_id + s + str(corr_end) + '_' + str(s_len)
        sub_id = trim_contig_id + s + str(corr_start) + '_' + str(corr_end)


    else:
        f_id = contig_id + s + str(0) + '_' + str(r_int)
        r_id = contig_id + s + str(r_int + length) + '_' + str(s_len)
        sub_id = contig_id + s + str(r_int) + '_' + str(r_int + length)

    sub_seq_df = pd.DataFrame([[sub_id, sub_seq, len(sub_seq)]],
                              columns=['contig_id', 'seq', 'seq_len']
                              )
    lo_seq_df = pd.DataFrame([[f_id, f_seq, len(f_seq)], [r_id, r_seq, len(r_seq)]],
                             columns=['contig_id', 'seq', 'seq_len'])

    return sub_seq_df, lo_seq_df


def sel_contig(real_contig_df, sub_sag_df):
    synth_contig_list = []
    for contig_id, seq_len in zip(sub_sag_df['contig_id'], sub_sag_df['seq_len']):
        len_df = real_contig_df.loc[real_contig_df['seq_len'] >= seq_len]
        r_sample = len_df.sample()
        split_seqs_dfs = build_subseq(r_sample, seq_len)
        real_contig_df = real_contig_df.loc[
            ~real_contig_df['contig_id'].isin(list(r_sample['contig_id']))
        ]
        real_contig_df = pd.concat([real_contig_df, split_seqs_dfs[1]])
        synth_contig_list.append(split_seqs_dfs[0])
    concat_df = pd.concat(synth_contig_list)

    return concat_df


src_gen_path = sys.argv[1]  # path containing the reference genomes

src_gen_list = [os.path.join(src_gen_path, f) for f in os.listdir(src_gen_path)
                if (('.fa' in f) | ('.fna' in f) | ('.fasta' in f))
                ]
print(src_gen_list)
gen_dict = {}
for gen_fasta in src_gen_list:
    with open(gen_fasta, 'r') as fa_in:
        data = fa_in.read().split('>')
    rec_list = []
    for rec in data:
        if len(rec.split('\n', 1)) > 1:
            header, seq = rec.split('\n', 1)
            clear_header = header.split(' ', 1)[0].replace('|', '_')
            seq = seq.replace('\n', '')
            seq_len = len(seq)
            rec_list.append((clear_header, seq, seq_len))
    rec_df = pd.DataFrame(rec_list, columns=['contig_id', 'seq', 'seq_len'])
    gen_dict[gen_fasta] = rec_df

synth_sag_file = sys.argv[2]  # file containing SAGs ditributions, Synth-SAG_contig_length.tsv
synth_sag_df = pd.read_csv(synth_sag_file, header=0, sep='\t')

for gen_id in gen_dict.keys():
    print(gen_id)
    gen_recs_df = gen_dict[gen_id]
    gen_max_len = max(list(gen_recs_df['seq_len']))
    gen_sum_len = sum(list(gen_recs_df['seq_len']))
    gen_len_len = len(list(gen_recs_df['seq_len']))
    sag_id_list = list(synth_sag_df['sag_id'].unique())
    sag_counter = 0
    sag_done_list = []
    while sag_counter < 10:
        sag_rand = random.choices(sag_id_list)[0]
        sag_id_list.remove(sag_rand)
        sub_df = synth_sag_df[['sag_id', 'contig_id', 'seq_len', 'statistic', 'p_value'
                               ]].loc[synth_sag_df['sag_id'] == sag_rand
                                      ]
        sub_df.sort_values(by='seq_len', ascending=False, inplace=True)
        max_contig_len = max(list(sub_df['seq_len']))
        sum_contig_len = sum(list(sub_df['seq_len']))
        len_contig_len = len(list(sub_df['seq_len']))

        if max_contig_len <= gen_max_len:
            try:
                gen_synth_df = sel_contig(gen_recs_df, sub_df)
                sag_save_file = 'synthetic_SAGs/' + \
                                gen_id.rsplit('/', 1)[1].rsplit('.', 1)[0] + '.' + str(sag_rand) + \
                                '.fasta'
                with open(sag_save_file, 'w') as syn_fa:
                    fasta_list = ['>' + f[0] + '\n' + f[1] + '\n' for f in
                                  zip(gen_synth_df['contig_id'], gen_synth_df['seq'])
                                  ]
                    syn_fa.write(''.join(fasta_list))
                sag_counter += 1
            except:
                print('ERROR', max_contig_len, gen_max_len)
        else:
            print(max_contig_len, gen_max_len)
