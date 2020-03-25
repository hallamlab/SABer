import sys
import random
from Bio import SeqIO
import os

def get_SAGs(sag_path):
    # Find the SAGs!
    if os.path.isdir(sag_path):
        print('[SABer]: Directory specified, looking for SAGs\n')
        sag_list = [os.path.join(sag_path, f) for f in
                    os.listdir(sag_path) if ((f.split('.')[-1] == 'fasta' or
                    f.split('.')[-1] == 'fna') and 'Sample' not in f)
                    ]
        print('[SABer]: Found %s SAGs in directory\n'
                     % str(len(sag_list))
                     )

    elif os.path.isfile(sag_path):
        print('[SABer]: File specified, processing %s\n'
                     % os.path.basename(sag_path)
                     )
        sag_list = [sag_path]

    return sag_list


def build_subcontigs(in_fasta, subcontig_path, max_contig_len, overlap_len):
    basename = os.path.basename(in_fasta)
    samp_id = basename.rsplit('.', 1)[0]
    contigs = get_seqs(in_fasta)
    headers, subs = kmer_slide(contigs, max_contig_len,
                                        overlap_len
                                        )

    return (samp_id, headers, subs)


def kmer_slide(seq_list, n, o_lap):
    all_sub_seqs = []
    all_sub_headers = []
    for seq_tup in seq_list:
        header, seq = seq_tup
        clean_seq = seq.strip('\n').lower()
        sub_list = get_frags(clean_seq, n, o_lap)
        sub_headers = [header + '_' + str(i) for i, x in
                        enumerate(sub_list, start=0)
                        ]
        all_sub_seqs.extend(sub_list)
        all_sub_headers.extend(sub_headers)

    return all_sub_headers, all_sub_seqs


def get_frags(seq, l_max, o_lap):
    "Fragments the seq into subseqs of length l_max and overlap of o_lap."
    "Leftover tail overlaps with tail-1"
    "Currently, if a seq is < l_max, it returns the full seq"
    seq_frags = []
    if (l_max != 0) and (len(seq) > l_max):
        offset = l_max - o_lap
        for i in range(0, len(seq), offset):
            if i+l_max < len(seq):
                frag = seq[i:i+l_max]
                seq_frags.append(frag)
            else:
                frag = seq[i:]
                seq_frags.append(frag)
                break
    else:
        seq_frags.append(seq)

    return seq_frags


def get_seqs(fasta_file):
    sag_contigs = []
    with open(fasta_file, 'r') as fasta_in:
        for record in SeqIO.parse(fasta_in, 'fasta'): # TODO: replace biopython with base python
            f_id = record.id
            #f_description = record.description
            f_seq = str(record.seq)
            if f_seq != '':
                sag_contigs.append((f_id, f_seq))

    return sag_contigs



def main(sag_path, save_path, max_contig_len, min_contig_len, overlap_len, per_comp):
    max_contig_len = int(max_contig_len)
    min_contig_len = int(min_contig_len)
    overlap_len = int(overlap_len)
    per_comp = int(per_comp)
    # Find the SAGs!
    sag_list = get_SAGs(sag_path)
    # Build subcontiges for SAGs
    sag_subcontigs = [build_subcontigs(sag, save_path,
                                               max_contig_len, overlap_len
                                               ) for sag in sag_list]

    for s in sag_subcontigs:
        sag_id, sag_headers, sag_subs = s
        sag_full_len = sum(len(x) for x in sag_subs)
        sag_per_comp = sag_full_len*(per_comp/100)
        zip_list = list(zip(sag_headers, [len(x) for x in sag_subs], sag_subs))
        small_list = [[x[0], x[2]] for x in zip_list if min_contig_len <= x[1] < max_contig_len]
        big_list = [[x[0], x[2]] for x in zip_list if x[1] >= max_contig_len]
        random.Random(42).shuffle(small_list)
        mock_list = []
        for m in small_list:
            mock_comp_len = sum(len(z[1]) for z in mock_list)
            if mock_comp_len < sag_per_comp:
                mock_list.append(m)
        if mock_comp_len < sag_per_comp:
            random.Random(42).shuffle(big_list)
            for m in big_list:
                mock_comp_len = sum(len(z[1]) for z in mock_list)
                if mock_comp_len < sag_per_comp:
                    mock_list.append(m)

        mock_per_comp = sum(len(z[1]) for z in mock_list)
        mock_per_min = min(len(z[1]) for z in mock_list)
        mock_per_max = max(len(z[1]) for z in mock_list)
        mock_per_mean = sum(len(z[1]) for z in mock_list)/len(mock_list)
        print(sag_id, sag_full_len, sag_per_comp, len(mock_list), mock_per_comp,
                mock_per_min, mock_per_max, mock_per_mean
                )

        with open(os.path.join(save_path, sag_id +
                  '.mock.' + str(per_comp) + '.fasta'), 'w') as sub_out:
                sub_rec_list = ['\n'.join(['>'+rec[0], rec[1]])
                                for rec in mock_list
                                ]
                sub_out.write('\n'.join(sub_rec_list) + '\n')



if __name__ == '__main__':
    sag_path = '/home/rmclaughlin/Ryan/CAMI_I_HIGH/source_genomes/'
    save_path = '~/Ryan/test_SABer/mockSAGs/new_mocks/'
    max_contig_len = 8000
    min_contig_len = 1500
    overlap_len = 0
    per_comp = 40


    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
