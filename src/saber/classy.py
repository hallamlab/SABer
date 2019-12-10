__author__ = 'Ryan J McLaughlin'


import saber.utilities as s_utils


class SABerBase:
    """
    A base class for all SABer sub-commands. It requires shared properties
    """
    def __init__(self, subcmd_name) -> None:
        self.subcmd = subcmd_name
        self.executables = {}
        self.aln_file = ""
        self.seq_file = ""
        self.output_sep = ','
        return

    def get_info(self) -> str:
        info_string = ""
        return info_string

    def furnish_with_arguments(self, args) -> None:
        self.executables["bwa"] = s_utils.which("bwa")
        return

