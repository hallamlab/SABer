__author__ = 'Ryan J McLaughlin'

import glob
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from itertools import product, islice

import dit
import hdbscan
import numpy as np
import pandas as pd
import pyfastx
import umap
from dit.other import renyi_entropy
from skbio.stats.composition import clr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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
        n_spaces = 12 - len(exe)
        versions_string += "\t" + exe + ' ' * n_spaces + versions_dict[exe] + "\n"

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


def check_out_dirs(save_path, autoopt, mode):
    """Checks if dirs all exist in save_path, makes them if not.

    :param save_path: directory where all intermediate and final files are saved.
    :return: A dictionary with the stage dir and the full path."""

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #if autoopt == 'algo_defaults':
    #    mode_path = autoopt
    #else:
    mode_path = os.path.join(autoopt, mode)
    denovo_path = os.path.join(mode_path, 'denovo')
    hdbscan_path = os.path.join(mode_path, 'hdbscan')
    ocsvm_path = os.path.join(mode_path, 'ocsvm')
    inter_path = os.path.join(mode_path, 'intersect')
    xpgs_path = os.path.join(mode_path, 'xpgs')
    sd_list = [autoopt, mode_path, denovo_path,
               hdbscan_path, ocsvm_path, inter_path,
               xpgs_path
               ]
    sd_dict = {}
    for sd in sd_list:
        sd_path = os.path.join(save_path, sd)
        if not os.path.exists(sd_path):
            os.makedirs(sd_path)
        sd_key = os.path.basename(sd)
        sd_dict[sd_key] = sd_path

    return sd_dict


def get_SAGs(sag_path):
    # Find the SAGs!
    if os.path.isdir(sag_path):
        logging.info('Directory specified, looking for Trusted Contigs\n')
        sag_list = [os.path.join(sag_path, f) for f in
                    os.listdir(sag_path) if ((f.split('.')[-1] == 'fasta' or
                                              f.split('.')[-1] == 'fna' or
                                              f.split('.')[-1] == 'fa') and 'Sample' not in f)
                    ]
        logging.info('Found %s Trusted Contig files in directory\n'
                     % str(len(sag_list))
                     )

    elif os.path.isfile(sag_path):
        logging.info('File specified, processing %s\n'
                     % os.path.basename(sag_path)
                     )
        sag_list = [sag_path]
    else:
        pass  # TODO: add error exception for bad file path

    return sag_list


def build_subcontigs(seq_type, in_fasta_list, subcontig_path, max_contig_len, overlap_len, min_len):
    sub_list = []
    logging.info('\rLoading/Building subcontigs for {}\n'.format(seq_type))
    for in_fasta in tqdm(in_fasta_list):
        basename = os.path.basename(in_fasta)
        samp_id = basename.rsplit('.', 1)[0]
        sub_file = os.path.join(subcontig_path, samp_id + '.subcontigs.fasta')
        if os.path.exists(os.path.join(subcontig_path, samp_id + '.subcontigs.fasta')) == False:
            # get contigs from fasta file
            contigs = get_seqs(in_fasta)
            headers, subs = kmer_slide(contigs, int(max_contig_len),
                                       int(overlap_len), int(min_len)
                                       )
            if len(subs) != 0:
                with open(sub_file, 'w') as sub_out:
                    sub_out.write('\n'.join(['\n'.join(['>' + rec[0], rec[1]]) for rec in
                                             zip(headers, subs)]) + '\n'
                                  )
                sub_list.append((samp_id, sub_file))
        else:
            sub_list.append((samp_id, sub_file))

    logging.info('\n')
    sub_list = tuple(sub_list)
    return sub_list


def kmer_slide(scd_db, n, o_lap, m_len):
    all_sub_seqs = []
    all_sub_headers = []
    for k in scd_db:
        rec = k
        header, seq = rec[0], rec[1]
        if len(str(seq)) >= int(o_lap):
            clean_seq = str(seq).upper()
            sub_list = sliding_window(clean_seq, n, o_lap)
            sub_headers = [header + '_' + str(i) for i, x in
                           enumerate(sub_list, start=0)
                           ]
            all_sub_seqs.extend(sub_list)
            all_sub_headers.extend(sub_headers)
        elif len(str(seq)) >= int(m_len):
            all_sub_seqs.extend([seq])
            all_sub_headers.extend([header + '_0'])
    return tuple(all_sub_headers), tuple(all_sub_seqs)


def sliding_window(seq, win_size, o_lap):
    "Fragments the seq into subseqs of length win_size and overlap of o_lap."
    "Leftover tail overlaps with tail-1"
    "Currently, if a seq is < win_size, it returns the full seq"
    seq_frags = []
    # Verify the inputs
    try:
        it = iter(seq)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(win_size) == type(0)) and (type(o_lap) == type(0))):
        raise Exception("**ERROR** type(win_size) and type(win_size) must be int.")
    if o_lap > win_size:
        raise Exception("**ERROR** step must not be larger than win_size.")
    if win_size <= len(seq):
        i = 0
        offset = len(seq) - win_size
        while i + win_size <= offset:
            seq_frags.append(seq[i:i + win_size])
            i = i + win_size - o_lap
        seq_frags.append(seq[-win_size:])
    elif win_size > len(seq):
        seq_frags.append(seq)

    return seq_frags


def slidingWindow(sequence, winSize, step):
    # pulled source from https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/
    seq_frags = []
    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize <= len(sequence):
        numOfChunks = ((len(sequence) - winSize) // step) + 1
        for i in range(0, numOfChunks * step, step):
            seq_frags.append(sequence[i:i + winSize])
        seq_frags.append(sequence[-winSize:])  # add the remaining tail
    elif winSize > len(sequence):
        seq_frags.append(sequence)

    return seq_frags


def get_seqs(fasta_file):
    fasta = pyfastx.Fasta(fasta_file, build_index=False)

    return fasta


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


def tetra_cnt(fasta):  # TODO: add multi-processing to this function
    # Dict of all tetramers
    tetra_cnt_dict = {''.join(x): [] for x in product('atgc', repeat=4)}
    header_list = []
    # count up all tetramers and also populate the tetra dict
    subcontig_len_dict = {}
    for rec in fasta:
        header = rec[0]
        header_list.append(header)
        seq = rec[1]
        subcontig_len_dict[header] = len(seq)
        tmp_dict = {k: 0 for k, v in tetra_cnt_dict.items()}
        clean_seq = seq.strip('\n').lower()
        kmer_list = [''.join(x) for x in get_kmer(clean_seq, 4)]
        tetra_counter = Counter(kmer_list)
        # add counter to tmp_dict
        for tetra in tmp_dict.keys():
            count_tetra = int(tetra_counter[tetra])
            tmp_dict[tetra] = count_tetra
        # map tetras to their reverse tetras (not compliment)
        dedup_dict = {}
        for tetra in tmp_dict.keys():
            if (tetra not in dedup_dict.keys()) & (tetra[::-1] not in dedup_dict.keys()):
                dedup_dict[tetra] = ''
            elif tetra[::-1] in dedup_dict.keys():
                dedup_dict[tetra[::-1]] = tetra
        # combine the tetras and their reverse (not compliment)
        tetra_prop_dict = {}
        for tetra in dedup_dict.keys():
            if dedup_dict[tetra] != '':
                tetra_prop_dict[tetra] = tmp_dict[tetra] + tmp_dict[dedup_dict[tetra]]
            else:
                tetra_prop_dict[tetra] = tmp_dict[tetra]
        # add to tetra_cnt_dict
        for k in tetra_cnt_dict.keys():
            if k in tetra_prop_dict.keys():
                tetra_cnt_dict[k].append(tetra_prop_dict[k])
            else:
                tetra_cnt_dict[k].append(0.0)
    # convert the final dict into a pd dataframe for ease
    tetra_cnt_dict['contig_id'] = header_list
    tetra_cnt_df = pd.DataFrame.from_dict(tetra_cnt_dict).set_index('contig_id')
    dedupped_df = tetra_cnt_df.loc[:, (tetra_cnt_df != 0.0).any(axis=0)]
    dedupped_df += 1  # TODO: adds pseudo-count, is there a better way?
    first_val = dedupped_df.columns[0]
    last_val = dedupped_df.columns[-1]
    dedupped_df['sum'] = dedupped_df.sum(axis=1)
    # Covert to proportion
    prop_df = dedupped_df.loc[:, first_val:last_val].div(dedupped_df['sum'], axis=0)
    # Normalize proportions to length of subcontig
    normal_list = [prop_df.loc[i] / subcontig_len_dict[i] for i in subcontig_len_dict.keys()]
    normal_df = pd.DataFrame(normal_list, columns=prop_df.columns, index=prop_df.index)
    # Transform using CLR
    clr_df = normal_df.apply(clr)
    # Standardize the mg tetra DF
    scale = StandardScaler().fit(clr_df.values)  # TODO this should be added to the tetra_cnt step
    scaled_data = scale.transform(clr_df.values)
    std_tetra_df = pd.DataFrame(scaled_data, index=clr_df.index)

    return std_tetra_df


def check_ent(ent, skip_list):
    skip = False
    for s in skip_list:
        if s in ent:
            skip = True
    return skip


def runCleaner(dir_path, ptrn, skip_list=[]):
    ptrn_glob = glob.glob(os.path.join(dir_path, ptrn))
    for ent in ptrn_glob:
        if check_ent(ent, skip_list) == False:
            if os.path.isfile(ent):
                try:
                    os.remove(ent)
                except:
                    print("Error while deleting file : ", ent)
            elif os.path.isdir(ent):
                try:
                    shutil.rmtree(ent)
                except:
                    print("Error while deleting directory : ", ent)


##########################################################################
# Below is a work in progress for the entropy/param matching for AutoOpt #
##########################################################################
def set_clust_params(denovo_min_clust, denovo_min_samp, anchor_min_clust,
                     anchor_min_samp, nu, gamma, vr, r, s, vs, a, abund_file,
                     working_dir
                     ):
    logging.info('Running AutoOpt to find optimal hyperparameters\n')
    clust_match_df = calc_entropy(working_dir, [abund_file])
    autoopt_method, autoopt_setting, autoopt_params = run_param_match(working_dir, a, vr, r, s, vs)  # TODO: draw from dev_utils/param_matching.py
    logging.info('AutoOpt method: ' + str(autoopt_method) + '\n')
    logging.info('Parameter set: ' + autoopt_params['setting'] + '\n')
    logging.info('\tDe Novo min_cluster_size: ' + str(autoopt_params['d_min_clust']) + '\n')
    logging.info('\tDe Novo min_samples: ' + str(autoopt_params['d_min_samp']) + '\n')
    logging.info('\tAnchored min_cluster_size: ' + str(autoopt_params['a_min_clust']) + '\n')
    logging.info('\tAnchored min_samples: ' + str(autoopt_params['a_min_samp']) + '\n')
    logging.info('\tAnchored nu: ' + str(autoopt_params['nu']) + '\n')
    logging.info('\tAnchored gamma: ' + str(autoopt_params['gamma']) + '\n')

    return autoopt_method, autoopt_setting, autoopt_params


def entropy_cluster(ent_df):
    logging.disable(logging.DEBUG)
    samp2type = {x: y for x, y in zip(ent_df['sample_id'], ent_df['sample_type'])}
    piv_df = ent_df.pivot(index='sample_id', columns='alpha', values='Renyi_Entropy')
    scaler = StandardScaler()
    scale_fit = scaler.fit(piv_df)
    scale_emb = scale_fit.transform(piv_df)
    scale_df = pd.DataFrame(scale_emb, index=piv_df.index.values,
                            columns=piv_df.columns
                            )
    umap_fit = umap.UMAP(n_neighbors=2, min_dist=0.0, n_components=2,
                         random_state=42
                         ).fit(scale_df)
    umap_emb = umap_fit.transform(scale_df)
    umap_df = pd.DataFrame(umap_emb, index=scale_df.index.values, columns=['u0', 'u1'])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, allow_single_cluster=False,
                                prediction_data=True
                                ).fit(umap_df)

    cluster_labels = clusterer.labels_
    cluster_probs = clusterer.probabilities_
    cluster_outlier = clusterer.outlier_scores_

    umap_df['sample_type'] = [samp2type[x] for x in umap_df.index.values]
    umap_df['sample_id'] = umap_df.index.values
    umap_df['cluster'] = cluster_labels
    umap_df['probabilities'] = cluster_probs
    umap_df['outlier_scores'] = cluster_outlier

    ent_umap_df = ent_df.merge(umap_df, on=['sample_id', 'sample_type'], how='left')
    ent_best_df = find_best_match(piv_df, ent_umap_df)

    return umap_df, ent_best_df, piv_df, umap_fit, clusterer, scale_fit


def find_best_match(piv_df, ent_umap_df):
    # Closest ref sample methods
    outlier_list = list(ent_umap_df.query("cluster == -1")['sample_id'])
    cmpr_list = []
    for r1, row1 in piv_df.iterrows():
        keep_diff = [r1, '', np.inf]
        for r2, row2 in piv_df.iterrows():
            euc_d = np.linalg.norm(row1 - row2)
            if euc_d < keep_diff[2] and r1 != r2 and r2 not in outlier_list:
                keep_diff = [r1, r2, euc_d]
        cmpr_list.append(keep_diff)
    cmpr_df = pd.DataFrame(cmpr_list, columns=['sample_id', 'best_match', 'euc_d'])
    ent_best_df = ent_umap_df.merge(cmpr_df, on='sample_id', how='left')

    return ent_best_df


def real_best_match(piv_df, real_piv_df, real_umap_df, working_dir):
    # Closest ref sample methods
    logging.disable(logging.DEBUG)
    r_cmpr_list = []
    for r1, row1 in real_piv_df.iterrows():
        keep_diff = [r1, '', np.inf]
        for r2, row2 in piv_df.iterrows():
            euc_d = np.linalg.norm(row1 - row2)
            if euc_d < keep_diff[2]:
                keep_diff = [r1, r2, euc_d]
        r_cmpr_list.append(keep_diff)
    r_cmpr_df = pd.DataFrame(r_cmpr_list, columns=['sample_id', 'best_match', 'euc_d'])
    best_df = real_umap_df.merge(r_cmpr_df, on='sample_id', how='left')
    best_df.to_csv(os.path.join(working_dir, 'cluster_table.tsv'), sep='\t', index=False)
    return best_df


def real_cluster(clusterer, real_df, umap_fit, scale_fit):
    logging.disable(logging.DEBUG)
    # Assign real data to clusters
    real_piv_df = real_df.pivot(index='sample_id', columns='alpha', values='Renyi_Entropy')
    scale_emb = scale_fit.transform(real_piv_df)
    scale_df = pd.DataFrame(scale_emb, index=real_piv_df.index.values, columns=real_piv_df.columns)
    umap_emb = umap_fit.transform(scale_df)
    umap_df = pd.DataFrame(umap_emb, index=scale_df.index.values, columns=['u0', 'u1'])
    test_labels, strengths = hdbscan.approximate_predict(clusterer, umap_df)
    cluster_labels = test_labels
    cluster_probs = strengths
    samp2type = {x: y for x, y in zip(real_df['sample_id'], real_df['sample_type'])}
    umap_df['sample_type'] = [samp2type[x] for x in umap_df.index.values]
    umap_df['sample_id'] = umap_df.index.values
    umap_df['cluster'] = cluster_labels
    umap_df['probabilities'] = cluster_probs
    return real_piv_df, umap_df


def calc_real_entrophy(mba_cov_list, working_dir):
    logging.disable(logging.DEBUG)
    logging.info('Calculating Renyi Entropy profile\n')
    entropy_list = []
    for samp_file in mba_cov_list:
        samp_id = samp_file.split('/')[-1].rsplit('.', 1)[0]
        samp_label = samp_id
        samp_rep = 0
        cov_df = pd.read_csv(samp_file, sep='\t', header=0)
        if cov_df.shape[0] > 100000:  # need to sub-sample if table is huge
            cov_df = cov_df.sample(n=int(100000), random_state=42, replace=False)
        cov_df['hash_id'] = [hashlib.sha256(x.encode(encoding='utf-8')).hexdigest()
                             for x in cov_df['contigName']
                             ]
        depth_sum = cov_df['totalAvgDepth'].sum()
        cov_df['relative_depth'] = [x / depth_sum for x in cov_df['totalAvgDepth']]
        cov_dist = dit.Distribution(cov_df['hash_id'].tolist(),
                                    cov_df['relative_depth'].tolist()
                                    )
        q_list = [0, 1, 2, 4, 8, 16, 32, np.inf]
        for q in tqdm(q_list):
            r_ent = renyi_entropy(cov_dist, q)
            entropy_list.append([samp_id, samp_label, samp_rep, q, r_ent])
    real_df = pd.DataFrame(entropy_list, columns=['sample_id', 'sample_type',
                                                  'sample_rep', 'alpha',
                                                  'Renyi_Entropy'
                                                  ])
    # Have to replace the np.inf with a real value for plotting
    real_df['alpha_int'] = real_df['alpha'].copy()
    real_df['alpha_int'].replace(4, 3, inplace=True)
    real_df['alpha_int'].replace(8, 4, inplace=True)
    real_df['alpha_int'].replace(16, 5, inplace=True)
    real_df['alpha_int'].replace(32, 6, inplace=True)
    real_df['alpha_int'].replace(np.inf, 7, inplace=True)
    x_labels = {0: 'Richness (a=0)', 1: 'Shannon (a=1)',
                2: 'Simpson (a=2)', 3: '4', 4: '8', 5: '16',
                6: '32', 7: 'Berger–Parker (a=inf)'
                }
    real_df['x_labels'] = [x_labels[x] for x in real_df['alpha_int']]
    real_df.to_csv(os.path.join(working_dir, 'entropy_table.tsv'), sep='\t', index=False)

    return real_df


def remove_outliers(ent_best_df, real_merge_df, umap_fit, scale_fit):
    logging.disable(logging.DEBUG)
    # If labeled as an outlier, take the closest match
    keep_cols = ['sample_id', 'sample_type', 'alpha', 'Renyi_Entropy', 'alpha_int',
                 'x_labels', 'u0', 'u1', 'cluster', 'probabilities', 'best_match', 'euc_d'
                 ]
    best_merge_df = pd.concat([ent_best_df[keep_cols],
                               real_merge_df[keep_cols]]
                              )
    # Calculate centroids for cluster
    real_id_list = real_merge_df['sample_id']
    cent_merge_df = calc_centroid(best_merge_df,
                                  best_merge_df[['sample_id', 'alpha',
                                                 'Renyi_Entropy',
                                                 'cluster']],
                                  real_id_list, umap_fit, scale_fit
                                  )

    return cent_merge_df


def calc_centroid(best_df, ren_df, real_ids, umap_fit, scale_fit):
    logging.disable(logging.DEBUG)
    ren_piv_df = ren_df.pivot(index=['sample_id', 'cluster'], columns='alpha', values='Renyi_Entropy').reset_index()
    sample_ren_df = ren_piv_df.query('cluster == -1').drop(['cluster'], axis=1)
    sample_ren_df.set_index('sample_id', inplace=True)
    samp_scale_emb = scale_fit.transform(sample_ren_df)
    samp_scale_df = pd.DataFrame(samp_scale_emb, index=sample_ren_df.index.values, columns=sample_ren_df.columns)
    sample_ren_emb = umap_fit.transform(samp_scale_df)
    sample_umap_df = pd.DataFrame(sample_ren_emb, index=samp_scale_df.index.values, columns=['u0', 'u1'])
    cluster_ren_df = ren_piv_df.query('cluster != -1 and sample_id not in @real_ids'
                                      ).drop(['sample_id'], axis=1)
    cluster_ren_df.set_index('cluster', inplace=True)
    clust_scale_emb = scale_fit.transform(cluster_ren_df)
    clust_scale_df = pd.DataFrame(clust_scale_emb, index=cluster_ren_df.index.values, columns=cluster_ren_df.columns)
    cluster_ren_emb = umap_fit.transform(clust_scale_df)
    cluster_umap_df = pd.DataFrame(cluster_ren_emb, index=clust_scale_df.index.values, columns=['u0', 'u1'])
    median_df = cluster_umap_df.groupby(by=cluster_umap_df.index).median()
    out_dict = {}
    for r1, row1 in sample_umap_df.iterrows():
        keep_diff = [r1, '', np.inf]
        for r2, row2 in median_df.iterrows():
            euc_d = np.linalg.norm(row1 - row2)
            if euc_d < keep_diff[2] and r1 != r2:
                keep_diff = [r1, r2, euc_d]
        out_dict[r1] = keep_diff
    best_df['best_cluster'] = [out_dict[x][1] if x in
                                                 out_dict.keys() else y for x, y
                               in zip(best_df['sample_id'],
                                      best_df['cluster']
                                      )]

    return best_df


def calc_entropy(working_dir, mba_cov_list):
    real_clean = os.path.join(working_dir, 'cluster_clean.tsv')
    if os.path.isfile(real_clean):
        logging.info('Entropy profile exists, moving on...\n')
        real_only_df = pd.read_csv(real_clean, sep='\t', header=0)
    else:
        logging.info('Loading Reference Renyi Entropy profiles\n')
        ent_file = os.path.join(os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0],
                                'saber/configs/entropy_table.tsv'
                                )
        ent_df = pd.read_csv(ent_file, sep='\t', header=0)
        ent_results = entropy_cluster(ent_df)
        ent_best_df = ent_results[1]
        piv_df = ent_results[2]
        umap_fit = ent_results[3]
        clusterer = ent_results[4]
        scale_fit = ent_results[5]

        # Cluster real samples
        real_df = calc_real_entrophy(mba_cov_list, working_dir)
        real_piv_df, real_umap_df = real_cluster(clusterer, real_df, umap_fit, scale_fit)
        real_best_df = real_best_match(piv_df, real_piv_df, real_umap_df, working_dir)
        real_merge_df = real_df.merge(real_best_df, on=['sample_id', 'sample_type'], how='left')

        # Replace outliers with best match
        best_merge_df = remove_outliers(ent_best_df, real_merge_df, umap_fit, scale_fit)
        real_samp = real_df['sample_id'].values[0]
        # Save final files
        real_only_df = best_merge_df.query('sample_id == @real_samp')
        real_only_df.to_csv(real_clean, sep='\t', index=False)

    return real_only_df


#####################################################################################################################################################################################################################
def best_match_params(real_dir):
    clust_all_file = os.path.join(os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0],
                                  'saber/configs/CV_clust_table.tsv'
                                  )
    clust_all_df = pd.read_csv(clust_all_file, sep='\t', header=0)
    real_df = pd.read_csv(os.path.join(real_dir, 'cluster_clean.tsv'), sep='\t', header=0)
    real_df = real_df[['sample_type', 'sample_id', 'best_match', 'best_cluster']]
    # best_match
    nc_max_df = clust_all_df.groupby(['sample_id', 'cv_algo', 'algo', 'level',
                                      'cv_param1', 'cv_param2', 'cv_val1', 'cv_val2']
                                     )[['nc_cnt']].max().reset_index()
    mq_max_df = clust_all_df.groupby(['sample_id', 'cv_algo', 'algo', 'level',
                                      'cv_param1', 'cv_param2', 'cv_val1', 'cv_val2']
                                     )[['mq_cnt']].max().reset_index()
    nc_hdb_df = nc_max_df.query("cv_algo == 'hdbscan' & (algo == 'hdbscan' | algo == 'denovo')")
    nc_ocs_df = nc_max_df.query("cv_algo == 'ocsvm' & algo == 'ocsvm'")
    mq_hdb_df = mq_max_df.query("cv_algo == 'hdbscan' & (algo == 'hdbscan' | algo == 'denovo')")
    mq_ocs_df = mq_max_df.query("cv_algo == 'ocsvm' & algo == 'ocsvm'")
    nc_hdb_sort_df = nc_hdb_df.sort_values(['nc_cnt', 'cv_val1', 'cv_val2'],
                                           ascending=[False, True, True]
                                           )
    nc_ocs_sort_df = nc_ocs_df.sort_values(['nc_cnt', 'cv_val1', 'cv_val2'],
                                           ascending=[False, False, True]
                                           )
    mq_hdb_sort_df = mq_hdb_df.sort_values(['mq_cnt', 'cv_val1', 'cv_val2'],
                                           ascending=[False, True, True]
                                           )
    mq_ocs_sort_df = mq_ocs_df.sort_values(['mq_cnt', 'cv_val1', 'cv_val2'],
                                           ascending=[False, False, True]
                                           )
    nc_hdb_dup_df = nc_hdb_sort_df.drop_duplicates(subset=['sample_id', 'cv_algo',
                                                           'algo', 'level']
                                                   )
    nc_ocs_dup_df = nc_ocs_sort_df.drop_duplicates(subset=['sample_id', 'cv_algo',
                                                           'algo', 'level']
                                                   )
    mq_hdb_dup_df = mq_hdb_sort_df.drop_duplicates(subset=['sample_id', 'cv_algo',
                                                           'algo', 'level']
                                                   )
    mq_ocs_dup_df = mq_ocs_sort_df.drop_duplicates(subset=['sample_id', 'cv_algo',
                                                           'algo', 'level']
                                                   )
    best_nc_hdb_df = real_df.merge(nc_hdb_dup_df, left_on='best_match',
                                   right_on='sample_id', how='left'
                                   )
    best_nc_ocs_df = real_df.merge(nc_ocs_dup_df, left_on='best_match',
                                   right_on='sample_id', how='left'
                                   )
    best_mq_hdb_df = real_df.merge(mq_hdb_dup_df, left_on='best_match',
                                   right_on='sample_id', how='left'
                                   )
    best_mq_ocs_df = real_df.merge(mq_ocs_dup_df, left_on='best_match',
                                   right_on='sample_id', how='left'
                                   )
    best_nc_hdb_df['mq_nc'] = 'nc'
    best_nc_ocs_df['mq_nc'] = 'nc'
    best_mq_hdb_df['mq_nc'] = 'mq'
    best_mq_ocs_df['mq_nc'] = 'mq'
    keep_cols = ['sample_type', 'sample_id_x', 'best_match', 'cv_algo', 'algo', 'level',
                 'cv_param1', 'cv_param2', 'cv_val1', 'cv_val2', 'mq_nc'
                 ]
    best_cat_df = pd.concat([best_nc_hdb_df[keep_cols], best_nc_ocs_df[keep_cols],
                             best_mq_hdb_df[keep_cols], best_mq_ocs_df[keep_cols]
                             ])
    best_dup_df = best_cat_df.drop_duplicates()
    best_dup_df.to_csv(os.path.join(real_dir, 'best_match_params.tsv'), sep='\t', index=False)

    return best_dup_df, real_df


def best_cluster_params(real_dir, real_df):
    nc_agg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0],
                               'saber/configs/NC_agg_params.tsv'
                               )
    nc_agg_df = pd.read_csv(nc_agg_file, sep='\t', header=0)
    mq_agg_file = os.path.join(os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0],
                               'saber/configs/MQ_agg_params.tsv'
                               )
    mq_agg_df = pd.read_csv(mq_agg_file, sep='\t', header=0)
    nc_clust_df = nc_agg_df.query("grouping == 'best_cluster'")
    mq_clust_df = mq_agg_df.query("grouping == 'best_cluster'")
    nc_clust_df['group_val'] = nc_clust_df['group_val'].astype(int)
    mq_clust_df['group_val'] = mq_clust_df['group_val'].astype(int)
    nc_max_df = nc_clust_df.groupby(['group_val', 'cv_algo', 'algo', 'level',
                                     'cv_param1', 'cv_param2', 'cv_val1', 'cv_val2']
                                    )['nc_cnt'].max().reset_index()
    mq_max_df = mq_clust_df.groupby(['group_val', 'cv_algo', 'algo', 'level',
                                     'cv_param1', 'cv_param2', 'cv_val1', 'cv_val2']
                                    )['mq_cnt'].max().reset_index()
    nc_hdb_df = nc_max_df.query("cv_algo == 'hdbscan' & (algo == 'hdbscan' | algo == 'denovo')")
    nc_ocs_df = nc_max_df.query("cv_algo == 'ocsvm' & algo == 'ocsvm'")
    mq_hdb_df = mq_max_df.query("cv_algo == 'hdbscan' & (algo == 'hdbscan' | algo == 'denovo')")
    mq_ocs_df = mq_max_df.query("cv_algo == 'ocsvm' & algo == 'ocsvm'")
    nc_hdb_sort_df = nc_hdb_df.sort_values(['nc_cnt', 'cv_val1', 'cv_val2'],
                                           ascending=[False, True, True]
                                           )
    nc_ocs_sort_df = nc_ocs_df.sort_values(['nc_cnt', 'cv_val1', 'cv_val2'],
                                           ascending=[False, False, True]
                                           )
    mq_hdb_sort_df = mq_hdb_df.sort_values(['mq_cnt', 'cv_val1', 'cv_val2'],
                                           ascending=[False, True, True]
                                           )
    mq_ocs_sort_df = mq_ocs_df.sort_values(['mq_cnt', 'cv_val1', 'cv_val2'],
                                           ascending=[False, False, True]
                                           )
    nc_hdb_dup_df = nc_hdb_sort_df.drop_duplicates(subset=['group_val', 'cv_algo',
                                                           'algo', 'level']
                                                   )
    nc_ocs_dup_df = nc_ocs_sort_df.drop_duplicates(subset=['group_val', 'cv_algo',
                                                           'algo', 'level']
                                                   )
    mq_hdb_dup_df = mq_hdb_sort_df.drop_duplicates(subset=['group_val', 'cv_algo',
                                                           'algo', 'level']
                                                   )
    mq_ocs_dup_df = mq_ocs_sort_df.drop_duplicates(subset=['group_val', 'cv_algo',
                                                           'algo', 'level']
                                                   )
    best_nc_hdb_df = real_df.merge(nc_hdb_dup_df, left_on='best_cluster',
                                   right_on='group_val', how='left'
                                   )
    best_nc_ocs_df = real_df.merge(nc_ocs_dup_df, left_on='best_cluster',
                                   right_on='group_val', how='left'
                                   )
    best_mq_hdb_df = real_df.merge(mq_hdb_dup_df, left_on='best_cluster',
                                   right_on='group_val', how='left'
                                   )
    best_mq_ocs_df = real_df.merge(mq_ocs_dup_df, left_on='best_cluster',
                                   right_on='group_val', how='left'
                                   )
    best_nc_hdb_df['mq_nc'] = 'nc'
    best_nc_ocs_df['mq_nc'] = 'nc'
    best_mq_hdb_df['mq_nc'] = 'mq'
    best_mq_ocs_df['mq_nc'] = 'mq'
    keep_cols = ['sample_type', 'sample_id', 'best_cluster', 'cv_algo', 'algo', 'level',
                 'cv_param1', 'cv_param2', 'cv_val1', 'cv_val2', 'mq_nc'
                 ]
    best_cat_df = pd.concat([best_nc_hdb_df[keep_cols], best_nc_ocs_df[keep_cols],
                             best_mq_hdb_df[keep_cols], best_mq_ocs_df[keep_cols]
                             ])
    best_dup_df = best_cat_df.drop_duplicates()
    best_dup_df.to_csv(os.path.join(real_dir, 'best_cluster_params.tsv'), sep='\t', index=False)

    return best_dup_df, nc_agg_df, mq_agg_df


def majority_rule_params(real_dir, nc_agg_df, mq_agg_df, real_df):
    nc_major_df = nc_agg_df.query("grouping == 'majority_rule'")
    mq_major_df = mq_agg_df.query("grouping == 'majority_rule'")
    nc_max_df = nc_major_df.groupby(['group_val', 'cv_algo', 'algo', 'level',
                                     'cv_param1', 'cv_param2', 'cv_val1', 'cv_val2']
                                    )['nc_cnt'].max().reset_index()
    mq_max_df = mq_major_df.groupby(['group_val', 'cv_algo', 'algo', 'level',
                                     'cv_param1', 'cv_param2', 'cv_val1', 'cv_val2']
                                    )['mq_cnt'].max().reset_index()
    nc_hdb_df = nc_max_df.query("cv_algo == 'hdbscan' & (algo == 'hdbscan' | algo == 'denovo')")
    nc_ocs_df = nc_max_df.query("cv_algo == 'ocsvm' & algo == 'ocsvm'")
    mq_hdb_df = mq_max_df.query("cv_algo == 'hdbscan' & (algo == 'hdbscan' | algo == 'denovo')")
    mq_ocs_df = mq_max_df.query("cv_algo == 'ocsvm' & algo == 'ocsvm'")
    nc_hdb_sort_df = nc_hdb_df.sort_values(['nc_cnt', 'cv_val1', 'cv_val2'],
                                           ascending=[False, True, True]
                                           )
    nc_ocs_sort_df = nc_ocs_df.sort_values(['nc_cnt', 'cv_val1', 'cv_val2'],
                                           ascending=[False, False, True]
                                           )
    mq_hdb_sort_df = mq_hdb_df.sort_values(['mq_cnt', 'cv_val1', 'cv_val2'],
                                           ascending=[False, True, True]
                                           )
    mq_ocs_sort_df = mq_ocs_df.sort_values(['mq_cnt', 'cv_val1', 'cv_val2'],
                                           ascending=[False, False, True]
                                           )
    nc_hdb_dup_df = nc_hdb_sort_df.drop_duplicates(subset=['group_val', 'cv_algo',
                                                           'algo', 'level']
                                                   )
    nc_ocs_dup_df = nc_ocs_sort_df.drop_duplicates(subset=['group_val', 'cv_algo',
                                                           'algo', 'level']
                                                   )
    mq_hdb_dup_df = mq_hdb_sort_df.drop_duplicates(subset=['group_val', 'cv_algo',
                                                           'algo', 'level']
                                                   )
    mq_ocs_dup_df = mq_ocs_sort_df.drop_duplicates(subset=['group_val', 'cv_algo',
                                                           'algo', 'level']
                                                   )
    nc_hdb_dup_df['tmp_key'] = 0
    nc_ocs_dup_df['tmp_key'] = 0
    mq_hdb_dup_df['tmp_key'] = 0
    mq_ocs_dup_df['tmp_key'] = 0
    real_df['tmp_key'] = 0
    best_nc_hdb_df = real_df.merge(nc_hdb_dup_df, on='tmp_key', how='left')
    best_nc_ocs_df = real_df.merge(nc_ocs_dup_df, on='tmp_key', how='left')
    best_mq_hdb_df = real_df.merge(mq_hdb_dup_df, on='tmp_key', how='left')
    best_mq_ocs_df = real_df.merge(mq_ocs_dup_df, on='tmp_key', how='left')
    best_nc_hdb_df['mq_nc'] = 'nc'
    best_nc_ocs_df['mq_nc'] = 'nc'
    best_mq_hdb_df['mq_nc'] = 'mq'
    best_mq_ocs_df['mq_nc'] = 'mq'
    keep_cols = ['sample_type', 'sample_id', 'cv_algo', 'algo', 'level',
                 'cv_param1', 'cv_param2', 'cv_val1', 'cv_val2', 'mq_nc'
                 ]
    best_cat_df = pd.concat([best_nc_hdb_df[keep_cols], best_nc_ocs_df[keep_cols],
                             best_mq_hdb_df[keep_cols], best_mq_ocs_df[keep_cols]
                             ])
    best_dup_df = best_cat_df.drop_duplicates()
    best_dup_df.to_csv(os.path.join(real_dir, 'majority_rule_params.tsv'), sep='\t', index=False)

    return best_dup_df


def run_param_match(real_dir, autoopt_setting, vr, r, s, vs):
    # Need to get best params for best_match, best_cluster, and majority_rule
    # Assign the best match params to the SI data
    best_match_df, real_df = best_match_params(real_dir)
    # best_cluster
    best_cluster_df, nc_agg_df, mq_agg_df = best_cluster_params(real_dir, real_df)
    # majority_rule
    majority_rule_df = majority_rule_params(real_dir, nc_agg_df, mq_agg_df, real_df)
    opt_param_dict = {'majority_rule': majority_rule_df,
                      'best_cluster': best_cluster_df,
                      'best_match': best_match_df,
                      'algo_defaults': None
                      }
    if ((autoopt_setting != 'algo_defaults') & 
          (vr == None and r == None and s == None and vs == None)
          ):
        opt_df = opt_param_dict[autoopt_setting]
        vs = 'very_strict'
    elif autoopt_setting != 'algo_defaults':
        opt_df = opt_param_dict[autoopt_setting]
    elif ((autoopt_setting == 'algo_defaults') & 
          (vr != None or r != None or s != None or vs != None)
          ):
        autoopt_setting = 'majority_rule'
        opt_df = opt_param_dict[autoopt_setting]
    if vr:  # TODO: this can be refactored, and should be at some point
        d_hdb_df = opt_df.query("cv_algo == 'hdbscan' & algo == 'denovo'"
                                "& mq_nc == 'mq' & level == 'strain'"
                                )
        a_hdb_df = opt_df.query("cv_algo == 'hdbscan' & algo == 'hdbscan'"
                                "& mq_nc == 'mq' & level == 'strain'"
                                )
        ocs_df = opt_df.query("cv_algo == 'ocsvm' & algo == 'ocsvm'"
                              "& mq_nc == 'mq' & level == 'strain'"
                              )
        setting = 'very_relaxed'
    elif r:
        d_hdb_df = opt_df.query("cv_algo == 'hdbscan' & algo == 'denovo'"
                                "& mq_nc == 'mq' & level == 'exact'"
                                )
        a_hdb_df = opt_df.query("cv_algo == 'hdbscan' & algo == 'hdbscan'"
                                "& mq_nc == 'mq' & level == 'exact'"
                                )
        ocs_df = opt_df.query("cv_algo == 'ocsvm' & algo == 'ocsvm'"
                              "& mq_nc == 'mq' & level == 'exact'"
                              )
        setting = 'relaxed'

    elif s:
        d_hdb_df = opt_df.query("cv_algo == 'hdbscan' & algo == 'denovo'"
                                "& mq_nc == 'nc' & level == 'strain'"
                                )
        a_hdb_df = opt_df.query("cv_algo == 'hdbscan' & algo == 'hdbscan'"
                                "& mq_nc == 'nc' & level == 'strain'"
                                )
        ocs_df = opt_df.query("cv_algo == 'ocsvm' & algo == 'ocsvm'"
                              "& mq_nc == 'nc' & level == 'strain'"
                              )
        setting = 'strict'

    elif vs:
        d_hdb_df = opt_df.query("cv_algo == 'hdbscan' & algo == 'denovo'"
                                "& mq_nc == 'nc' & level == 'exact'"
                                )
        a_hdb_df = opt_df.query("cv_algo == 'hdbscan' & algo == 'hdbscan'"
                                "& mq_nc == 'nc' & level == 'exact'"
                                )
        ocs_df = opt_df.query("cv_algo == 'ocsvm' & algo == 'ocsvm'"
                              "& mq_nc == 'nc' & level == 'exact'"
                              )
        setting = 'very_strict'

    else:  # else use algo defaults
        setting = 'Default'
        params_dict = {'d_min_clust': 5, 'd_min_samp': None,
                       'a_min_clust': 5, 'a_min_samp': None,
                       'nu': 0.5, 'gamma': 'scale', 'setting': setting
                       }

    if setting != 'Default':
        opt_d_min_clust = int(d_hdb_df['cv_val1'].values[0])
        opt_d_min_samp = int(d_hdb_df['cv_val2'].values[0])
        opt_a_min_clust = int(a_hdb_df['cv_val1'].values[0])
        opt_a_min_samp = int(a_hdb_df['cv_val2'].values[0])
        opt_nu = float(ocs_df['cv_val1'].values[0])
        if ocs_df['cv_val2'].values[0] == 'scale':
            opt_gamma = ocs_df['cv_val2'].values[0]
        else:
            opt_gamma = float(ocs_df['cv_val2'].values[0])

        params_dict = {'d_min_clust': opt_d_min_clust, 'd_min_samp': opt_d_min_samp,
                       'a_min_clust': opt_a_min_clust, 'a_min_samp': opt_a_min_samp,
                       'nu': opt_nu, 'gamma': opt_gamma, 'setting': setting
                       }
    
    return autoopt_setting, setting, params_dict
