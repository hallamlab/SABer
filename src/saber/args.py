__author__ = 'Ryan J McLaughlin'

import argparse


class SABerArgumentParser(argparse.ArgumentParser):
    """
    A base argparse ArgumentParser for SABer with functions to furnish with common arguments.
    This standardizes the interface for a unified aesthetic across all sub-commands
    """

    def __init__(self, **kwargs):
        """
        Instantiate the argparse argument-parser and create three broad argument groups:
            reqs - for the required parameters
            optopt - for the optional parameters
            miscellany - for the miscellaneous parameters that are module agnostic,
            for example verbose, help, num_threads
        :param kwargs:
        """
        super(SABerArgumentParser, self).__init__(add_help=False, **kwargs)
        self.reqs = self.add_argument_group("Required parameters")
        self.seqops = self.add_argument_group("Sequence operation arguments")
        self.optopt = self.add_argument_group("Optional options")
        self.miscellany = self.add_argument_group("Miscellaneous options")

        self.miscellany.add_argument("-v", "--verbose", action="store_true", default=False,
                                     help="Prints a more verbose runtime log")
        self.miscellany.add_argument("-h", "--help",
                                     action="help",
                                     help="Show this help message and exit")

    def parse_args(self, args=None, namespace=None):
        args = super(SABerArgumentParser, self).parse_args(args=args, namespace=namespace)

        return args

    def add_recruit_args(self):
        self.reqs.add_argument("-s", "--sag", required=True, dest="sag_path",
                               help="Path to reference SAG FASTA file or directory "
                                    "containing only FASTA files."
                               )
        self.reqs.add_argument("-m", "--metag", required=True, dest="mg_file",
                               help="Path to a metagenome assembly [FASTA format only]."
                               )
        self.reqs.add_argument("-l", "--metaraw", required=True, dest="mg_raw_file_list",
                               help="Text file containing paths to raw FASTQ files for samples. "
                                    "One file per line, supports interleaved and separate PE reads. "
                                    "For separate PE files, both file paths on one line sep by [tab]."
                               )
        self.reqs.add_argument("-o", "--output-dir", required=True, dest="save_path",
                               help="Path to directory for all outputs."
                               )
        self.optopt.add_argument("--max_contig_len", required=False, default=10000,
                                 dest="max_contig_len",
                                 help="Max subcontig length in basepairs [10000]."
                                 )
        self.optopt.add_argument("--overlap_len", required=False, default=2000,
                                 dest="overlap_len",
                                 help="subcontig overlap in basepairs [2000]."
                                 )
        self.optopt.add_argument("--minhash_jacc_thresh", required=False, default=0.01,
                                 dest="jacc_thresh",
                                 help="Minimum Jaccard index between SAG and MG "
                                      "subcontig MinHash signatures [0.01]."
                                 )
        self.optopt.add_argument("--minhash_pass_prop", required=False, default=0.01,
                                 dest="mh_per_pass",
                                 help="Proportion of subcontigs required to pass "
                                      "MinHash signatures [0.01]."
                                 )
        self.optopt.add_argument("--abund_pass_prop", required=False, default=0.51,
                                 dest="abund_per_pass",
                                 help="Proportion of subcontigs required to pass "
                                      "Abundance filter [0.51]."
                                 )
        self.optopt.add_argument("--tetra_pass_prop", required=False, default=0.01,
                                 dest="gmm_per_pass",
                                 help="Proportion of subcontigs required to pass "
                                      "tetranucleotide filter [0.51]."
                                 )
        self.optopt.add_argument("-t", "--num_threads", required=False, default=1,
                                 dest="nthreads",
                                 help="Number of threads [1]."
                                 )
        self.optopt.add_argument("--force", required=False, default=False,
                                 action="store_true",
                                 help="Force SABer to run even if final recruits files exist [False]"
                                 )
        return
