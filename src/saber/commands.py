__author__ = 'Ryan J McLaughlin'

import logging
import os

from pip._internal.operations import freeze

# import saber
import saber.abundance_recruiter as abr
import saber.classy as s_class
import saber.clusterer as clst
import saber.compile_recruits as com
import saber.logger as s_log
import saber.minhash_recruiter as mhr
import saber.s_args as s_args
import saber.tetranuc_recruiter as tra
import saber.utilities as s_utils
from saber.__init__ import version


def info(sys_args):
    """
    Function for writing version information about SABer and python dependencies.
    Other related info (citation, executable versions, etc.) should also be written through this sub-command.
    Create a SABerBase object for the `info` sub-command

    :param sys_args: List of arguments parsed from the command-line.
    :return: None
    """
    parser = s_args.SABerArgumentParser(description="Return package and executable information.")
    args = parser.parse_args(sys_args)
    s_log.prep_logging()
    info_s = s_class.SABerBase("info")

    logging.info("SABer version " + version + ".\n")

    # Write the version of all python deps
    py_deps = {x.split('==')[0]: x.split('==')[1] for x in freeze.freeze()}

    logging.info("Python package dependency versions:\n\t" +
                 "\n\t".join([k + ": " + v for k, v in py_deps.items()]) + "\n")

    # Write the version of executable deps
    info_s.furnish_with_arguments(args)
    logging.info(s_utils.executable_dependency_versions(info_s.executables))  # TODO: needs updating for SABer exe

    if args.verbose:  # TODO: look at TS to determine what this is for.
        pass
        # logging.info(summary_str)

    return


def recruit(sys_args):
    """

    :param sys_args: List of arguments parsed from the command-line.
    :return: None
    """
    parser = s_args.SABerArgumentParser(description="Recruit environmental reads to reference contigs.")
    parser.add_recruit_args()
    args = parser.parse_args(sys_args)

    recruit_s = s_class.SABerBase("recruit")
    recruit_s.trust_path = args.trust_path
    recruit_s.mg_file = args.mg_file
    recruit_s.mg_raw_file_list = args.mg_raw_file_list
    recruit_s.pacbio = args.pacbio
    recruit_s.save_path = args.save_path
    recruit_s.max_contig_len = int(args.max_contig_len)
    recruit_s.overlap_len = int(args.overlap_len)
    recruit_s.min_len = int(args.min_len)
    recruit_s.kmer_size = int(args.kmer_size)
    recruit_s.jaccard = float(args.jaccard)
    recruit_s.nthreads = int(args.nthreads)
    recruit_s.force = args.force
    # Collect args for clustering
    recruit_s.denovo_min_clust, recruit_s.denovo_min_samp = args.denovo_min_clust, args.denovo_min_samp
    recruit_s.anchor_min_clust, recruit_s.anchor_min_samp = args.anchor_min_clust, args.anchor_min_samp
    recruit_s.nu, recruit_s.gamma = args.nu, args.gamma
    recruit_s.vr, recruit_s.r = args.vr_params, args.r_params
    recruit_s.s, recruit_s.vs = args.s_params, args.vs_params
    recruit_s.a = args.auto_params

    # Quick check to see which mode was selected
    mode_list = [recruit_s.vr, recruit_s.r, recruit_s.s, recruit_s.vs]
    recruit_s.mode = 'algo_defaults'
    for m in mode_list:  # TODO: this needs to break if more than one mode is selected
        if m:
            recruit_s.mode = m
    s_log.prep_logging(os.path.join(recruit_s.save_path, "SABer_log.txt"), verbosity=args.verbose)

    # TODO: should check for prebuilt files before anything else to avoid rebuilding things
    # TODO: think about setting a default upper and lower bp size for bins to filter bad ones

    # Build subcontigs for MG
    mg_file = tuple([os.path.splitext(os.path.basename(recruit_s.mg_file))[0],
                     recruit_s.mg_file])  # TODO: needs to support multiple MetaGs
    mg_sub_file = s_utils.build_subcontigs('Metagenomes', [recruit_s.mg_file],
                                           recruit_s.save_path,
                                           recruit_s.max_contig_len,
                                           recruit_s.overlap_len,
                                           recruit_s.min_len
                                           )[0]

    # Build minhash signatures if there are trusted contigs
    if recruit_s.trust_path:
        # Find the Trusted Contigs (TCs)
        tc_list = s_utils.get_SAGs(recruit_s.trust_path)
        trust_files = tuple([(os.path.splitext(os.path.basename(x))[0], x) for x in tc_list])
        trust_subs = s_utils.build_subcontigs('SAGs', tc_list,
                                                   recruit_s.save_path,
                                                   recruit_s.max_contig_len,
                                                   recruit_s.overlap_len,
                                                   recruit_s.min_len
                                                   )

        # Run MinHash recruiting algorithm
        minhash_df_dict = mhr.run_minhash_recruiter(recruit_s.save_path,  # TODO: expose some params for users
                                                    recruit_s.save_path,
                                                    trust_subs, mg_file,
                                                    recruit_s.nthreads,
                                                    recruit_s.min_len,
                                                    recruit_s.kmer_size
                                                    )
    else:
        minhash_df_dict = False
        trust_subs = tuple()

    # Build abundance tables
    abund_scale_file, abund_raw_file = abr.runAbundRecruiter(recruit_s.save_path,
                                                             recruit_s.save_path, mg_sub_file,
                                                             recruit_s.mg_raw_file_list,
                                                             recruit_s.pacbio, recruit_s.nthreads
                                                             )
    # Build tetra hz tables
    tetra_file = tra.run_tetra_recruiter(recruit_s.save_path,
                                         mg_sub_file
                                         )
    # Set all clustering parameters
    recruit_s.mode, recruit_s.set, recruit_s.params_dict = s_utils.set_clust_params(recruit_s.denovo_min_clust,
                                                                                  recruit_s.denovo_min_samp,
                                                                                  recruit_s.anchor_min_clust,
                                                                                  recruit_s.anchor_min_samp,
                                                                                  recruit_s.nu, recruit_s.gamma,
                                                                                  recruit_s.vr, recruit_s.r,
                                                                                  recruit_s.s, recruit_s.vs,
                                                                                  recruit_s.a, abund_raw_file,
                                                                                  recruit_s.save_path
                                                                                  )

    # Build save dir structure and start logging file
    save_dirs_dict = s_utils.check_out_dirs(recruit_s.save_path, recruit_s.mode, recruit_s.set)

    # Run HDBSCAN Cluster and Trusted Cluster Cleaning
    mg_id = mg_sub_file[0]
    clusters = clst.runClusterer(mg_id, save_dirs_dict[recruit_s.set], save_dirs_dict[recruit_s.set],
                                 abund_scale_file, tetra_file,
                                 minhash_df_dict,
                                 recruit_s.params_dict['d_min_clust'],
                                 recruit_s.params_dict['d_min_samp'],
                                 recruit_s.params_dict['a_min_clust'],
                                 recruit_s.params_dict['a_min_samp'],
                                 recruit_s.params_dict['nu'],
                                 recruit_s.params_dict['gamma'],
                                 recruit_s.jaccard,
                                 recruit_s.nthreads
                                 )
    # Collect and join all recruits
    com.run_combine_recruits(save_dirs_dict, recruit_s.mg_file,
                             clusters, trust_files, recruit_s.set,
                             recruit_s.nthreads
                             )

    return
