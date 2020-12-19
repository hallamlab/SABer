import argparse
import logging
from os.path import isfile, basename
from os.path import join as o_join

import pandas as pd
import saber.logger as s_log
import saber.utilities as s_utils
from sklearn import svm


class tetra_recruiter:

    def __init__(self, tra_path, sag_sub_files, mg_sub_file, rpkm_max_df, gmm_per_pass):
        self.tra_path = tra_path
        self.sag_sub_files = sag_sub_files
        self.mg_sub_file = mg_sub_file
        self.rpkm_max_df = rpkm_max_df
        self.gmm_per_pass = gmm_per_pass

        self.mg_id = self.mg_sub_file[0]
        self.predictors = ['ocsvm', 'gmm']
        # self.total_pass_lists = dict.fromkeys(self.predictors, [])
        self.ml_functions = {'ocsvm': self.runOCSVM, 'gmm': self.runGMM}  # will be replace with ML package

    def run_tetra_recruiter(self):
        logging.info('1')

        # tetra files always exist
        self.mg_tetra_df, self.mg_headers = self.loadMg()

        # iterate through all sags
        for i, sag_rec in enumerate(self.sag_sub_files):

            sag_id, sag_headers, sag_subs, paths = self.loadSag(sag_rec)

            if all([isfile(x) for x in paths]):
                pass_lists = self.loadPassLists(sag_id, paths)
            else:
                pass_lists = self.calcPassLists(sag_id, sag_subs, sag_headers)

            # This piece of sag now has a pass_list 
            self.storePassLists(sag_id, pass_lists)

        return

    def calcPassLists(self, sag_id, sag_subs, sag_headers):
        logging.info('2')
        sag_tetra_df = self.loadSagTetra(sag_id, sag_subs, sag_headers)
        mg_tetra_filter_df, mg_rpkm_contig_list = self.concatSagMg(sag_id)
        pass_lists = self.Train(sag_id, sag_tetra_df, mg_tetra_filter_df, mg_rpkm_contig_list)

        # for pred_name in self.predictors:
        #     with open(o_join(self.tra_path, sag_id + '.'+ pred_name+'_recruits.tsv'), 'w') as tra_out:
        #         tra_out.write('\n'.join(['\t'.join(x) for x in pass_lists[pred_name]]))

        return pass_lists

    def Train(self, sag_id, sag_tetra_df, mg_tetra_filter_df, mg_rpkm_contig_list):
        logging.info('3')
        pass_lists = dict.fromkeys(self.predictors)
        for pred_name in self.predictors:
            passed_items = self.ml_functions[pred_name](sag_id, sag_tetra_df, mg_tetra_filter_df, mg_rpkm_contig_list)
            pass_lists[pred_name] = passed_items

        return pass_lists

    def runOCSVM(self, sag_id, sag_tetra_df, mg_tetra_filter_df, mg_rpkm_contig_list):
        logging.info('4')
        clf = svm.OneClassSVM()
        clf.fit(sag_tetra_df.values)
        sag_pred = clf.predict(sag_tetra_df.values)
        mg_pred = clf.predict(mg_tetra_filter_df.values)
        mg_pred_df = pd.DataFrame(data=mg_pred, index=mg_tetra_filter_df.index.values)

        svm_pass_df = mg_pred_df.loc[mg_pred_df[0] != -1]
        svm_pass_df = svm_pass_df.loc[svm_pass_df.index.isin(mg_rpkm_contig_list)]
        svm_pass_list = []
        for md_nm in svm_pass_df.index.values:
            svm_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])

        logging.info('[SABer]: Reccruited %s subcontigs to %s with OCSVM\n' % (len(svm_pass_list), sag_id))

        return svm_pass_list

    def runGMM(self, sag_id, sag_tetra_df, mg_tetra_filter_df, mg_rpkm_contig_list):
        logging.info('4')
        clf = svm.OneClassSVM()
        clf.fit(sag_tetra_df.values)
        sag_pred = clf.predict(sag_tetra_df.values)
        mg_pred = clf.predict(mg_tetra_filter_df.values)
        mg_pred_df = pd.DataFrame(data=mg_pred, index=mg_tetra_filter_df.index.values)

        svm_pass_df = mg_pred_df.loc[mg_pred_df[0] != -1]
        svm_pass_df = svm_pass_df.loc[svm_pass_df.index.isin(mg_rpkm_contig_list)]
        svm_pass_list = []
        for md_nm in svm_pass_df.index.values:
            svm_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])

        logging.info('[SABer]: Reccruited %s subcontigs to %s with OCSVM\n' % (len(svm_pass_list), sag_id))

        return svm_pass_list

        ########### Begin Helper Function #################

    def loadMg(self):
        logging.info('5')
        if isfile(o_join(self.tra_path, self.mg_id + '.tetras.tsv')):
            mg_tetra_df = pd.read_csv(o_join(self.tra_path, self.mg_id + '.tetras.tsv'), sep='\t', index_col=0,
                                      header=0)
            mg_headers = mg_tetra_df.index.values
        else:
            mg_tetra_df, mg_headers = calcMgTetra()

        return mg_tetra_df, mg_headers

    def calcMgTetra(self):
        logging.info('6')
        mg_subcontigs = s_utils.get_seqs(mg_sub_file)  # TODO: can this be removed?
        mg_headers = tuple(mg_subcontigs.keys())
        mg_subs = tuple([r.seq for r in mg_subcontigs])
        mg_tetra_df = s_utils.tetra_cnt(mg_subs)
        mg_tetra_df['contig_id'] = mg_headers
        mg_tetra_df.set_index('contig_id', inplace=True)
        mg_tetra_df.to_csv(o_join(tra_path, mg_id + '.tetras.tsv'), sep='\t')
        return mg_headers, mg_tetra_df

    def loadSag(self, sag_rec):
        logging.info('7')
        sag_id, sag_file = sag_rec
        sag_subcontigs = s_utils.get_seqs(sag_file)
        sag_headers = tuple(sag_subcontigs.keys())
        sag_subs = tuple([r.seq for r in sag_subcontigs])
        path = o_join(self.tra_path, sag_id + '.{}_recruits.tsv')
        paths = [path.format(x) for x in self.predictors]
        return sag_id, sag_headers, sag_subs, paths

    def loadSagTetra(self, sag_id, sag_subs, sag_headers):
        logging.info('8')
        if isfile(o_join(self.tra_path, sag_id + '.tetras.tsv')):
            sag_tetra_df = pd.read_csv(o_join(tra_path, sag_id + '.tetras.tsv'),
                                       sep='\t', index_col=0, header=0)
        else:
            sag_tetra_df = self.calcSagTetra(sag_id, sag_subs, sag_headers)

        return sag_tetra_df

    def calcSagTetra(self, sag_id, sag_subs, sag_headers):
        logging.info('9')
        sag_tetra_df = s_utils.tetra_cnt(sag_subs)
        sag_tetra_df['contig_id'] = sag_headers
        sag_tetra_df.set_index('contig_id', inplace=True)
        sag_tetra_df.to_csv(o_join(self, tra_path, sag_id + '.tetras.tsv'), sep='\t')
        return sag_tetra_df

    def loadPassLists(self, sag_id, paths):
        logging.info('10')
        pass_lists = dict.fromkeys(predictors, [])
        if all([isfile(x) for x in paths]):
            logging.info('[SABer]: Found recruit lists. Loading  %s tetramer Hz recruit list\n' % sag_id)
            for pred_name in self.predictors:
                with open(o_join(self.tra_path, sag_id + '.' + pred_name + '_recruits.tsv'), 'r') as tra_in:
                    pass_lists[pred_name] = [x.rstrip('\n').split('\t') for x in tra_in.readlines()]
        return pass_lists

    def concatSagMg(self, sag_id):
        logging.info('11')
        mg_rpkm_contig_list = list(self.rpkm_max_df.loc[self.rpkm_max_df['sag_id'] == sag_id]['subcontig_id'].values)
        mg_tetra_filter_df = self.mg_tetra_df.loc[self.mg_tetra_df.index.isin(mg_rpkm_contig_list)]
        return mg_tetra_filter_df, mg_rpkm_contig_list

    def storePassLists(self, sag_id, pass_lists):
        logging.info('12')
        all_pass_dict = dict.fromkeys(self.predictors)
        for pred_name in self.predictors:
            all_pass_dict[pred_name] = pd.DataFrame(pass_lists[pred_name],
                                                    columns=['sag_id', 'subcontig_id', 'contig_id'])

        for pred_name in self.predictors:
            all_pass_df = all_pass_dict[pred_name]
            mg_max_only_df = self.updateDF(all_pass_df, pred_name)

    def updateDF(self, all_pass_df, pred_name):
        logging.info('13')
        mg_tot_cnt_df = self.build_mg_tot_cnt()
        gmm_cnt_df = self.build_gmm_cnt(all_pass_df)
        df_output = gmm_cnt_df.merge(mg_tot_cnt_df, how='left', on='contig_id')

        df_output['percent_recruited'] = df_output['subcontig_recruits'] / \
                                         df_output['subcontig_total']
        df_output.sort_values(by='percent_recruited', ascending=False, inplace=True)
        df_output.to_csv(o_join(tra_path, mg_id + '.' + pred_name + '.check.tsv'), sep='\t', index=False)
        return df_output

    def build_mg_tot_cnt(self):
        logging.info('14')
        mg_contig_list = [x.rsplit('_', 1)[0] for x in self.mg_headers]
        mg_tot_cnt_df = pd.DataFrame(zip(mg_contig_list, self.mg_headers),
                                     columns=['contig_id', 'subcontig_id']).groupby(['contig_id']).count().reset_index()
        mg_tot_cnt_df.columns = ['contig_id', 'subcontig_total']
        return mg_tot_cnt_df

    def build_gmm_cnt(self, all_pass_df):
        logging.info('15')
        gmm_cnt_df = all_pass_df.groupby(['sag_id', 'contig_id']).count().reset_index()
        gmm_cnt_df.columns = ['sag_id', 'contig_id', 'subcontig_recruits']
        return gmm_cnt_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='uses tetrenucleotide Hz to recruit metaG reads to SAGs')
    parser.add_argument(
        '--tetra_path', help='path to tetrenucleotide output directory',
        required=True
    )
    parser.add_argument(
        '--sag_sub_file',
        help='path to SAG subcontigs file', required=True
    )
    # recruite from metagenomes
    parser.add_argument(
        '--mg_sub_file',
        help='path to metagenome subcontigs file', required=True
    )
    parser.add_argument(
        '--abund_df',
        help='path to output dataframe from abundance recruiter', required=True
    )
    parser.add_argument(
        '--per_pass',
        help='pass percentage of subcontigs to pass complete contig', required=True,
        default='0.01'
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Prints a more verbose runtime log"
                        )
    args = parser.parse_args()
    # set args
    tra_path = args.tetra_path
    sag_sub_file = args.sag_sub_file
    mg_sub_file = args.mg_sub_file
    abund_recruit_file = args.abund_df
    per_pass = float(args.per_pass)

    s_log.prep_logging("tetra_log.txt", args.verbose)
    sag_id = basename(sag_sub_file).rsplit('.', 2)[0]
    mg_id = basename(mg_sub_file).rsplit('.', 2)[0]
    abund_recruit_df = pd.read_csv(abund_recruit_file, header=0, sep='\t')
    logging.info('[SABer]: Starting Tetranucleotide Recruitment Step\n')

    tr = tetra_recruiter(tra_path, [[sag_id, sag_sub_file]], [mg_id, mg_sub_file],
                         abund_recruit_df, per_pass)

    tr.run_tetra_recruiter()
