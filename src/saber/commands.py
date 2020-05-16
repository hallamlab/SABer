__author__ = 'Ryan J McLaughlin'


from pip._internal.operations import freeze
import logging
import saber
import saber.args as s_args
import saber.classy as s_class
import saber.logger as s_log
import saber.utilities as s_utils
import saber.minhash_recruiter as mhr
import saber.abundance_recruiter as abr
import saber.tetranuc_recruiter as tra
import saber.compile_recruits as com

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

    logging.info("SABer version " + saber.version + ".\n")

    # Write the version of all python deps
    py_deps = {x.split('==')[0]:x.split('==')[1] for x in freeze.freeze()}


    logging.info("Python package dependency versions:\n\t" +
                 "\n\t".join([k + ": " + v for k, v in py_deps.items()]) + "\n")

    # Write the version of executable deps
    info_s.furnish_with_arguments(args)
    logging.info(s_utils.executable_dependency_versions(info_s.executables)) # TODO: needs updating for SABer exe

    if args.verbose: # TODO: look at TS to determine what this is for.
        pass
        # logging.info(summary_str)

    return


def recruit(sys_args):
    """

    :param sys_args: List of arguments parsed from the command-line.
    :return: None
    """
    parser = s_args.SABerArgumentParser(description="Recruit environmental reads to reference SAG(s).")
    parser.add_recruit_args()
    args = parser.parse_args(sys_args)

    s_log.prep_logging("SABer_log.txt", args.verbose)
    recruit_s = s_class.SABerBase("recruit")
    recruit_s.sag_path = args.sag_path
    recruit_s.mg_file = args.mg_file
    recruit_s.mg_raw_file_list = args.mg_raw_file_list
    recruit_s.save_path = args.save_path
    recruit_s.max_contig_len = args.max_contig_len
    recruit_s.overlap_len = args.overlap_len
    recruit_s.jacc_thresh = args.jacc_thresh
    recruit_s.abund_per_pass = args.abund_per_pass
    recruit_s.gmm_per_pass = args.gmm_per_pass
    recruit_s.mh_per_pass = args.mh_per_pass
    # Build save dir structure
    save_dirs_dict = s_utils.check_out_dirs(recruit_s.save_path)
    # Find the SAGs!
    sag_list = s_utils.get_SAGs(recruit_s.sag_path)

    # Build subcontiges for SAGs and MG
    sag_sub_files = [s_utils.build_subcontigs(sag,
                                               save_dirs_dict['subcontigs'],
                                               recruit_s.max_contig_len,
                                               recruit_s.overlap_len
                                               ) for sag in sag_list
                     ]
    mg_sub_file = s_utils.build_subcontigs(recruit_s.mg_file,
                                             save_dirs_dict['subcontigs'],
                                             recruit_s.max_contig_len,
                                             recruit_s.overlap_len
                                            )

    # Run MinHash recruiting algorithm
    logging.info('[SABer]: Starting Kmer Recruitment Step\n')
    minhash_df = mhr.run_minhash_recruiter(save_dirs_dict['signatures'],
    									   save_dirs_dict['minhash_recruits'],
                                           sag_sub_files, mg_sub_file,
                                           recruit_s.jacc_thresh, recruit_s.mh_per_pass
                                           )
    # Abundance Recruit Module
    logging.info('[SABer]: Starting Abundance Recruitment Step\n')
    abund_df = abr.run_abund_recruiter(save_dirs_dict['subcontigs'],
    								   save_dirs_dict['abund_recruits'], mg_subcontigs,
                                       recruit_s.mg_raw_file_list, minhash_df,
                                       recruit_s.abund_per_pass
                                       )
    # Tetranucleotide Hz Recruit Module
    logging.info('[SABer]: Starting Tetranucleotide Recruitment Step\n')
    tetra_df_dict = tra.run_tetra_recruiter(save_dirs_dict['tetra_recruits'],
    										sag_subcontigs, mg_subcontigs, abund_df,
                                       		recruit_s.gmm_per_pass
                                       		)
    # Collect and join all recruits
    logging.info('[SABer]: Combining All Recruits\n')
    mg_contigs = s_utils.get_seqs(recruit_s.mg_file)
    com.run_combine_recruits(save_dirs_dict['final_recruits'], save_dirs_dict['extend_SAGs'],
                             save_dirs_dict['re_assembled'], save_dirs_dict['checkM'],
                             mg_contigs, tetra_df_dict, minhash_df, mg_subcontigs, sag_list
                             )
    # Re-assemble SAG with MG recruits


    return
