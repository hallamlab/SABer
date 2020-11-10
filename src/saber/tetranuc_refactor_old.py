import logging
import saber.logger as s_log
import pandas as pd
import numpy as np
from os.path import isfile, basename
from os.path import join as o_join
import saber.utilities as s_utils
from sklearn.mixture import GaussianMixture as GMM
from sklearn import svm
from sklearn.ensemble import IsolationForest
import sys
import argparse
# from tetranuc_meta import MLModel



def run_tetra_recruiter(tra_path, sag_sub_files, mg_sub_file, rpkm_max_df, gmm_per_pass):
    predictors = ['ocsvm']
    ml_functions = {'ocsvm': runOCSVM}
    mg_id = mg_sub_file[0]
    
    ## if tetra files exist
    if isfile(o_join(tra_path, mg_id + '.tetras.tsv')):
        logging.info('[SABer]: Tetramer Hz matrix exist. Loading tetramer Hz matrix for %s\n' % mg_id)
        mg_tetra_df = pd.read_csv(o_join(tra_path, mg_id + '.tetras.tsv'), sep='\t', index_col=0, header=0)
        mg_headers = mg_tetra_df.index.values

    ## if tetra files don't exist
    else:
        logging.info('[SABer]: Tetramer Hz matrix not found. Calculating tetramer Hz matrix for %s\n' % mg_id)
        mg_subcontigs, mg_headers, mg_subs, mg_tetra_df = calculate_tetramer_hz_matrix(mg_id, mg_sub_file[1])

    ## pass list for all  
    total_pass_lists = dict.fromkeys(predictors, [])
    ###################################################

    #### for every sag
    for i, sag_rec in enumerate(sag_sub_files):
        sag_id, sag_file = sag_rec
        sag_subcontigs = s_utils.get_seqs(sag_file)
        sag_headers= tuple(sag_subcontigs.keys())
        sag_subs = tuple([r.seq for r in sag_subcontigs])

        path = o_join(tra_path, sag_id + '.{}_recruits.tsv')
        paths = [path.format(x) for x in predictors]
        
        ## record items that pass each predictors
        pass_lists = dict.fromkeys(predictors, [])

        ## load existing pass lists 
        if all([isfile(x) for x in paths]):
            logging.info('[SABer]: Found recruit lists. Loading  %s tetramer Hz recruit list\n' % sag_id)
            for pred_name in predictors:
                with open(o_join(tra_path, sag_id + '.' + pred_name + '_recruits.tsv'), 'r') as tra_in:
                    pass_lists[pred_name] = [x.rstrip('\n').split('\t') for x in tra_in.readlines()]
        
        ## Or create pass lists if not found
        else:
            logging.info('[SABer] Recruit lists not found.\n')
            
            if isfile(o_join(tra_path, sag_id + '.tetras.tsv')):
                logging.info('[SABer]: Loading tetramer Hz matrix for %s\n' % sag_id)
                sag_tetra_df = pd.read_csv(o_join(tra_path, sag_id + '.tetras.tsv'),
                                           sep='\t', index_col=0, header=0)
                logging.info('[SABer]: Loading tetramer Hz matrix for %s is completed\n' % sag_id)
            else:
                logging.info('[SABer]: Calculating tetramer Hz matrix for %s\n' % sag_id)
                sag_tetra_df = s_utils.tetra_cnt(sag_subs)
                sag_tetra_df['contig_id'] = sag_headers
                sag_tetra_df.set_index('contig_id', inplace=True)
                sag_tetra_df.to_csv(o_join(tra_path, sag_id + '.tetras.tsv'), sep='\t')

            # Concat SAGs and MG for GMM
            mg_rpkm_contig_list = list(rpkm_max_df.loc[rpkm_max_df['sag_id'] == sag_id]['subcontig_id'].values)
            outputdf = pd.DataFrame(data = mg_rpkm_contig_list)
            outputdf.to_csv(o_join(tra_path, '.list.tsv'), sep='\t', index=False)
            mg_tetra_filter_df = mg_tetra_df.loc[mg_tetra_df.index.isin(mg_rpkm_contig_list)]
            logging.info('[SABer]: Loading complete\n')
            logging.info('[SABer]: Now training ML\n')
            ## Calling function Train to handle all ML related work. It is standalone and is able to parallel process
            pass_lists = Train(rpkm_max_df, mg_tetra_df, sag_tetra_df, gmm_per_pass, mg_tetra_filter_df,mg_rpkm_contig_list, predictors, ml_functions)
            #########################################################################################################
            # for pred_name in predictors:
            #     with open(o_join(tra_path, sag_id + '.'+ pred_name+'_recruits.tsv'), 'w') as tra_out:
            #         tra_out.write('\n'.join(['\t'.join(x) for x in pass_lists[pred_name]]))
        
        for pred_name in predictors:
            total_pass_lists[pred_name] = pass_lists[pred_name]
            ##TODO: might be an error here
    ### end for every sag
    
    tetra_df_dict = dict.fromkeys(predictors)
    for pred_name in predictors:
        tetra_df_dict[pred_name] = pd.DataFrame(total_pass_lists[pred_name], columns=['sag_id', 'subcontig_id', 'contig_id'])

    
    for pred_name in tetra_df_dict:
        tetra_df = tetra_df_dict[pred_name]
        mg_max_only_df = updateDF(tetra_df, pred_name, mg_headers, gmm_per_pass)
        tetra_df_dict[pred_name] = mg_max_only_df
    
    return tetra_df_dict

def updateNewDF(tetra_df, tetra_id, mg_headers, gmm_per_pass):
    mg_tot_cnt_df = build_mg_tot_cnt(mg_headers)
    gmm_cnt_df = build_gmm_cnt(tetra_df)
    df_output = gmm_cnt_df.merge(mg_tot_cnt_df, how='left', on='contig_id')
    
    df_output['percent_recruited'] = df_output['subcontig_recruits'] / \
                                        df_output['subcontig_total']
    df_output.sort_values(by='percent_recruited', ascending=False, inplace=True)
    df_output.to_csv(o_join(tra_path, mg_id + '.' + pred_name + '.check.tsv'), sep='\t', index=False)
    return df_output

def build_mg_tot_cnt(mg_headers):
    mg_contig_list = [x.rsplit('_', 1)[0] for x in mg_headers]
    mg_tot_cnt_df = pd.DataFrame(zip(mg_contig_list, mg_headers),columns=['contig_id', 'subcontig_id']).groupby(['contig_id']).count().reset_index()
    mg_tot_cnt_df.columns = ['contig_id', 'subcontig_total']
    return mg_tot_cnt_df

def build_gmm_cnt(tetra_df):
    gmm_cnt_df = tetra_df.groupby(['sag_id', 'contig_id']).count().reset_index()
    gmm_cnt_df.columns = ['sag_id', 'contig_id', 'subcontig_recruits']
    return gmm_cnt_df

def updateDF(tetra_df, pred_name, mg_headers, gmm_per_pass):
    # logging.info(tetra_df)
    # logging.info('end\n')
    # logging.info(pred_name)
    # logging.info('end\n')
    # logging.info(mg_headers)
    # logging.info('end\n')
    # logging.info(len(mg_headers))
    
    # mg_headers_df = pd.DataFrame(data = mg_headers)
    # mg_headers_df.to_csv(o_join('.'  + '.mg_headers.tsv'), sep='\t', index=False)
    # logging.info("\nuploaded\n")

    gmm_cnt_df = tetra_df.groupby(['sag_id', 'contig_id']).count().reset_index()
    gmm_cnt_df.columns = ['sag_id', 'contig_id', 'subcontig_recruits']
    # Build subcontig count for each MG contig
    mg_contig_list = [x.rsplit('_', 1)[0] for x in mg_headers]
    logging.info(mg_contig_list)

    mg_tot_df = pd.DataFrame(zip(mg_contig_list, mg_headers),
                            columns=['contig_id', 'subcontig_id'])
    mg_tot_cnt_df = mg_tot_df.groupby(['contig_id']).count().reset_index()
    mg_tot_cnt_df.columns = ['contig_id', 'subcontig_total']
    mg_recruit_df = gmm_cnt_df.merge(mg_tot_cnt_df, how='left', on='contig_id')
    mg_recruit_df['percent_recruited'] = mg_recruit_df['subcontig_recruits'] / \
                                        mg_recruit_df['subcontig_total']
    mg_recruit_df.sort_values(by='percent_recruited', ascending=False, inplace=True)
    mg_recruit_df.to_csv(o_join(tra_path, mg_id + '.' + pred_name + '.check1.tsv'), sep='\t', index=False)


    # Only pass contigs that have the magjority of subcontigs recruited (>= N%)
    mg_recruit_filter_df = mg_recruit_df.loc[mg_recruit_df['percent_recruited'] >= float(gmm_per_pass)]
    mg_recruit_filter_df = mg_recruit_df
    mg_contig_per_max_df = mg_recruit_filter_df.groupby(['contig_id'])[
        'percent_recruited'].max().reset_index()
    mg_contig_per_max_df.columns = ['contig_id', 'percent_max']
    mg_recruit_max_df = mg_recruit_filter_df.merge(mg_contig_per_max_df, how='left',
                                                on='contig_id')
    mg_recruit_max_df.to_csv(o_join(tra_path, mg_id + '.' + pred_name + '.check2.tsv'), sep='\t', index=False)
    # Now pass contigs that have the maximum recruit % of subcontigs
    mg_max_only_df = mg_recruit_max_df.loc[mg_recruit_max_df['percent_recruited'] >=
                                        mg_recruit_max_df['percent_max']
                                        ]
    mg_max_only_df.to_csv(o_join(tra_path, mg_id + '.' + pred_name + '.tra_trimmed_recruits.tsv'), sep='\t', index=False)

    
    return mg_max_only_df
        

## This would become a caller for multithreading
def Train(rpkm_max_df, mg_tetra_df, sag_tetra_df, gmm_per_pass,mg_tetra_filter_df,mg_rpkm_contig_list, predictors, ml_functions):
    pass_lists = dict.fromkeys(predictors)
    for pred_name in predictors:
        passed_items = ml_functions[pred_name](rpkm_max_df, mg_tetra_df, sag_tetra_df, gmm_per_pass, mg_tetra_filter_df, mg_rpkm_contig_list)
        pass_lists[pred_name] = passed_items

    return pass_lists


def runOCSVM(rpkm_max_df, mg_tetra_df, sag_tetra_df, gmm_per_pass, mg_tetra_filter_df, mg_rpkm_contig_list):
    logging.info('[SABer]: Training OCSVM on SAG tetras\n')
    # fit OCSVM
    clf = svm.OneClassSVM()
    clf.fit(sag_tetra_df.values)
    sag_pred = clf.predict(sag_tetra_df.values)
    #sag_pred_df = pd.DataFrame(data=sag_pred, index=sag_tetra_df.index.values)
    mg_pred = clf.predict(mg_tetra_filter_df.values)
    ##############
    mg_tetra_filter_df.to_csv(o_join(tra_path, 'try_this' + mg_id + '.'  + '.mg_tetra_filter_df.tsv'), sep='\t')
    # mg_rpkm_contig_df = pd.DataFrame(data = mg_rpkm_contig_list)
    # mg_rpkm_contig_df.to_csv(o_join(tra_path, mg_id + '.'  + '.mg_rpkm.tsv'), sep='\t', index=False)
    # logging.info("output mg_tetra_filter_df")
    ####################
    mg_pred_df = pd.DataFrame(data=mg_pred, index=mg_tetra_filter_df.index.values)
    svm_pass_df = mg_pred_df.loc[mg_pred_df[0] != -1]
    # And is has to be from the RPKM pass list
    # logging.info("try here\n")
    # logging.info(mg_tetra_filter_df.index.values)
    # logging.info(mg_tetra_filter_df)
    # logging.info(sag_id)
    # logging.info(mg_pred_df)
    # logging.info(svm_pass_df)
    # logging.info("try_end\n")
    svm_pass_df = svm_pass_df.loc[svm_pass_df.index.isin(mg_rpkm_contig_list)]
    svm_pass_list = []
    for md_nm in svm_pass_df.index.values:
        svm_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])

    logging.info('[SABer]: Reccruited %s subcontigs to %s with OCSVM\n' % (len(svm_pass_list), sag_id))
    
    return svm_pass_list


def runGMM(rpkm_max_df, mg_tetra_df, sag_tetra_df, gmm_per_pass, predictors):
    # requires either coverage metrics or aic/bic
    return

def runIsoF(rpkm_max_df, mg_tetra_df, sag_tetra_df, gmm_per_pass, predictors):
    # requires coverage metrics
    logging.info('[SABer]: Training IsoForest on SAG tetras\n')
    clf = IsolationForest(n_estimators=100)
    return



def calculate_tetramer_hz_matrix(mg_id, mg_sub_file):
    logging.info('[SABer]: Calculating tetramer Hz matrix for %s\n' % mg_id)
    mg_subcontigs = s_utils.get_seqs(mg_sub_file) # TODO: can this be removed?
    mg_headers = tuple(mg_subcontigs.keys())
    mg_subs = tuple([r.seq for r in mg_subcontigs])
    mg_tetra_df = s_utils.tetra_cnt(mg_subs)
    mg_tetra_df['contig_id'] = mg_headers
    mg_tetra_df.set_index('contig_id', inplace=True)
    mg_tetra_df.to_csv(o_join(tra_path, mg_id + '.tetras.tsv'), sep='\t')
    return mg_subcontigs, mg_headers, mg_subs, mg_tetra_df




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
    run_tetra_recruiter(tra_path, [[sag_id, sag_sub_file]], [mg_id, mg_sub_file],
                        abund_recruit_df, per_pass)

