__author__ = 'Ryan J McLaughlin'

from Bio import SeqIO
import os
import re
import sys
import subprocess
import logging
from itertools import product, islice
from collections import Counter
import pandas as pd
import screed


def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def which(program):
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path_element in os.environ["PATH"].split(os.pathsep):
            path_element = path_element.strip('"')
            exe_file = os.path.join(path_element, program)
            if is_exe(exe_file):
                return exe_file
    return None


def executable_dependency_versions(exe_dict):
    """Function for retrieving the version numbers for each executable in exe_dict
    :param exe_dict: A dictionary mapping names of software to the path to their executable
    :return: A formatted string with the executable name and its respective version found"""
    versions_dict = dict()
    versions_string = "Software versions used:\n"

    simple_v = ["prodigal"]
    no_params = ["bwa"]
    version_re = re.compile(r"[Vv]\d+.\d|version \d+.\d|\d\.\d\.\d")

    for exe in exe_dict:
        ##
        # Get the help/version statement for the software
        ##
        versions_dict[exe] = ""
        if exe in simple_v:
            stdout, returncode = launch_write_command([exe_dict[exe], "-v"], True)
        elif exe in no_params:
            stdout, returncode = launch_write_command([exe_dict[exe]], True)
        else:
            logging.warning("Unknown version command for " + exe + ".\n")
            continue
        ##
        # Identify the line with the version number (since often more than a single line is returned)
        ##
        for line in stdout.split("\n"):
            if version_re.search(line):
                # If a line was identified, try to get just the string with the version number
                for word in line.split(" "):
                    if re.search(r"\d\.\d", word):
                        versions_dict[exe] = re.sub(r"[,:()[\]]", '', word)
                        break
                break
            else:
                pass
        if not versions_dict[exe]:
            logging.debug("Unable to find version for " + exe + ".\n")

    ##
    # Format the string with the versions of all software
    ##
    for exe in sorted(versions_dict):
        n_spaces = 12-len(exe)
        versions_string += "\t" + exe + ' '*n_spaces + versions_dict[exe] + "\n"

    return versions_string


def launch_write_command(cmd_list, just_do_it=False, collect_all=True):
    """Wrapper function for opening subprocesses through subprocess.Popen()

    :param cmd_list: A list of strings forming a complete command call
    :param just_do_it: Always return even if the returncode isn't 0
    :param collect_all: A flag determining whether stdout and stderr are returned
    via stdout or just stderr is returned leaving stdout to be written to the screen
    :return: A string with stdout and/or stderr text and the returncode of the executable"""
    stdout = ""
    if collect_all:
        proc = subprocess.Popen(cmd_list,
                                shell=False,
                                preexec_fn=os.setsid,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        stdout = proc.communicate()[0].decode("utf-8")
    else:
        proc = subprocess.Popen(cmd_list,
                                shell=False,
                                preexec_fn=os.setsid)
        proc.wait()

    # Ensure the command completed successfully
    if proc.returncode != 0 and not just_do_it:
        logging.error(cmd_list[0] + " did not complete successfully! Command used:\n" +
                      ' '.join(cmd_list) + "\nOutput:\n" + stdout)
        sys.exit(19)

    return stdout, proc.returncode


def check_out_dirs(save_path):
    """Checks if dirs all exist in save_path, makes them if not.

    :param save_path: directory where all intermediate and final files are saved.
    :return: A dictionary with the stage dir and the full path."""

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sd_list = ['subcontigs', 'signatures', 'minhash_recruits',
                     'abund_recruits', 'tetra_recruits', 'final_recruits',
                     'extend_SAGs', 're_assembled', 'checkM'
                     ]
    sd_dict = {}
    for sd in sd_list:
        sd_path = os.path.join(save_path, sd)
        if not os.path.exists(sd_path):
            os.makedirs(sd_path)
        sd_dict[sd] = sd_path

    return sd_dict


def get_SAGs(sag_path):
    # Find the SAGs!
    if os.path.isdir(sag_path):
        logging.info('[SABer]: Directory specified, looking for SAGs\n')
        sag_list = [os.path.join(sag_path, f) for f in
                    os.listdir(sag_path) if ((f.split('.')[-1] == 'fasta' or
                    f.split('.')[-1] == 'fna') and 'Sample' not in f)
                    ]
        logging.info('[SABer]: Found %s SAGs in directory\n'
                     % str(len(sag_list))
                     )

    elif os.path.isfile(sag_path):
        logging.info('[SABer]: File specified, processing %s\n'
                     % os.path.basename(sag_path)
                     )
        sag_list = [sag_path]

    return sag_list


def build_subcontigs(in_fasta, subcontig_path, max_contig_len, overlap_len):
    basename = os.path.basename(in_fasta)
    samp_id = basename.rsplit('.', 1)[0]
    logging.info('[SABer]: Loading/Building subcontigs for %s\n'
                 % samp_id
                 )

    sub_file = os.path.join(subcontig_path, samp_id + '.subcontigs.fasta')

    # Build sub sequences for all contigs
    '''
    if os.path.exists(os.path.join(subcontig_path, samp_id + '.subcontigs.fasta')) == True:
        contigs = get_seqs(os.path.join(subcontig_path, samp_id + '.subcontigs.fasta'))
        headers, subs = zip(*[contigs[k].name, contigs[k].sequence for k in contigs.keys()])
        headers = tuple(headers)
        subs = tuple(subs)
    '''
    #else:
    if os.path.exists(os.path.join(subcontig_path, samp_id + '.subcontigs.fasta')) == False:
        # get contigs from fasta file
        contigs = get_seqs(in_fasta)
        # remove any that are smaller that the max_contig_len
        #trim_contigs = [x for x in contigs if len(x[1]) >= int(max_contig_len)]
        headers, subs = kmer_slide(contigs, int(max_contig_len),
                                            int(overlap_len)
                                            )
        with open(sub_file, 'w') as sub_out:
                sub_out.write('\n'.join(['\n'.join(['>'+rec[0], rec[1]]) for rec in
                                zip(headers, subs)]) + '\n'
                             )

    return samp_id, sub_file
    #return (samp_id, headers, subs)


def kmer_slide(scd_db, n, o_lap):
    all_sub_seqs = []
    all_sub_headers = []
    for k in scd_db.keys():
        rec = scd_db[k]
        header, seq = rec.name, rec.sequence
        clean_seq = str(seq).upper()
        sub_list = slidingWindow(clean_seq, n, o_lap)
        sub_headers = [header + '_' + str(i) for i, x in
                        enumerate(sub_list, start=0)
                        ]
        all_sub_seqs.extend(sub_list)
        all_sub_headers.extend(sub_headers)


    return tuple(all_sub_headers), tuple(all_sub_seqs)


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
                frag = seq[-l_max:]
                seq_frags.append(frag)
                break
    #else:
    #    seq_frags.append(seq)

    return seq_frags


def slidingWindow(sequence, winSize, step): # pulled source from https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/

    seq_frags = []
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize <= len(sequence):
        numOfChunks = ((len(sequence)-winSize)//step)+1
        for i in range(0,numOfChunks*step,step):
            seq_frags.append(sequence[i:i+winSize])
        seq_frags.append(sequence[-winSize:]) # add the remaining tail
    else:
        seq_frags.append(sequence)

    return seq_frags


def get_seqs(fasta_file):

    if os.path.exists(fasta_file + '_screed') == False:
        screed.make_db(fasta_file)
    fadb = screed.ScreedDB(fasta_file)

    sag_contigs = []
    with open(fasta_file, 'r') as fasta_in:
        for record in SeqIO.parse(fasta_in, 'fasta'): # TODO: replace biopython with base python
            f_id = record.id
            f_description = record.description
            f_seq = str(record.seq)
            if f_seq != '':
                sag_contigs.append((f_id, f_seq))

    return fadb


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


def tetra_cnt(seq_list):
    # Dict of all tetramers
    tetra_cnt_dict = {''.join(x):[] for x in product('ATGC', repeat=4)}
    # count up all tetramers and also populate the tetra dict
    for seq in seq_list:
        tmp_dict = {k: 0 for k, v in tetra_cnt_dict.items()}
        clean_seq = seq.strip('\n').lower()
        kmer_list = [''.join(x) for x in get_kmer(clean_seq, 4)]
        tetra_counter = Counter(kmer_list)
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
        # combine the tetras and their reverse (not compliment), convert to proportions
        tetra_prop_dict = {}
        for tetra in dedup_dict.keys():
            if dedup_dict[tetra] != '':
                #tetra_prop_dict[tetra] = tmp_dict[tetra] + tmp_dict[dedup_dict[tetra]]
                t_prop = (tmp_dict[tetra]
                            + tmp_dict[dedup_dict[tetra]]) / total_kmer_cnt
                tetra_prop_dict[tetra] = t_prop
            else:
                #tetra_prop_dict[tetra] = tmp_dict[tetra]
                t_prop = tmp_dict[tetra] / total_kmer_cnt
                tetra_prop_dict[tetra] = t_prop
        # add to tetra_cnt_dict
        for k in tetra_cnt_dict.keys():
            if k in tetra_prop_dict.keys():
                tetra_cnt_dict[k].append(tetra_prop_dict[k])
            else:
                tetra_cnt_dict[k].append(0.0)
    # convert the final dict into a pd dataframe for ease
    tetra_cnt_df = pd.DataFrame.from_dict(tetra_cnt_dict)
    dedupped_df = tetra_cnt_df.loc[:, (tetra_cnt_df != 0.0).any(axis=0)]

    return dedupped_df

