import logging
from os.path import isfile, basename, getsize
from os.path import join as o_join
from subprocess import Popen

import pandas as pd

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import StandardScaler
import sys
import saber.utilities as s_utils


def runAbundRecruiter(subcontig_path, abr_path, mg_sub_file, mg_raw_file_list,
                      pacbio, nthreads
                      ):
    logging.info('Starting Abundance Data Transformation\n')
    mg_id = mg_sub_file[0]

    if isfile(o_join(abr_path, mg_id + '.coverage.scaled.tsv')):
        logging.info('Loading Abundance matrix for %s\n' % mg_id)
        mg_scale_out = o_join(abr_path, mg_id + '.coverage.scaled.tsv')
        mg_covm_out = o_join(abr_path, mg_id + '.mbacov.tsv')
    else:
        logging.info('Building %s abundance matrix\n' % mg_id)
        mg_sub_path = o_join(subcontig_path, mg_id + '.subcontigs.fasta')
        # Process raw metagenomes to calculate abundances
        mg_scale_out, mg_covm_out = procMetaGs(abr_path, mg_id, mg_raw_file_list,
                                               subcontig_path, pacbio, nthreads
                                               )
    # Clean up the directory
    logging.info('Cleaning up intermediate files...\n')
    for s in ["*.sam", "*.bam", "*.stderr.txt", "*.stdout.txt"]:
        s_utils.runCleaner(abr_path, s)

    return mg_scale_out, mg_covm_out


def procMetaGs(abr_path, mg_id, mg_raw_file_list, subcontig_path, pacbio, nthreads):
    # Process each raw metagenome
    with open(mg_raw_file_list, 'r') as raw_fa_in:
        raw_data = raw_fa_in.readlines()
    sam_list = []
    sorted_bam_list = []
    for line in raw_data:
        raw_file_list = line.strip('\n').split('\t')
        pe_id, mg_sam_out = runMiniMap2(abr_path, subcontig_path, mg_id, raw_file_list,
                                        pacbio, nthreads
                                        )
        sam_list.append(mg_sam_out)
        # Build/sorted .bam files
        mg_sort_out = runSamTools(abr_path, pe_id, nthreads, mg_id, mg_sam_out)
        sorted_bam_list.append(mg_sort_out)
        # Clean up intermediates
        logging.info('Cleaning up intermediate files...\n')
        for s in ["*.sam", "*.bam", "*.stderr.txt", "*.stdout.txt"]:
            s_utils.runCleaner(abr_path, s, skip_list=sorted_bam_list)



    logging.info('\n')
    mg_scale_out, mg_covm_out = runMBAcov(abr_path, mg_id, sorted_bam_list)
    # mg_covm_out = runCovM(abr_path, mg_id, nthreads, sorted_bam_list)
    # mg_covm_out = runSAMSAM(abr_path, subcontig_path, mg_id, sam_list, nthreads)
    # mg_covm_out = runPySAM(abr_path, subcontig_path, mg_id, sorted_bam_list, nthreads)
    return mg_scale_out, mg_covm_out


def runMiniMap2(abr_path, subcontig_path, mg_id, raw_file_list, pacbio, nthreads):
    pe1 = raw_file_list[0]
    if isfile(pe1) == True:
        pe_basename = basename(pe1)
        pe_id = pe_basename.split('.')[0]
        mg_sam_out = o_join(abr_path, pe_id + '.sam')
        try:  # if file exists but is empty
            sam_size = getsize(mg_sam_out)
        except:  # if file doesn't exist
            sam_size = -1
        if len(raw_file_list) == 2:
            logging.info('Raw reads in FWD and REV file...\n')
            pe2 = raw_file_list[1]
            mem_cmd = ['minimap2', '-ax', 'sr', '--split-prefix=tmp', '-t', str(nthreads), '-o', mg_sam_out,
                       o_join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1, pe2
                       ]
        elif ((len(raw_file_list) < 2) & (pacbio == True)):
            logging.info('Raw reads are PacBio Hifi...\n')
            mem_cmd = ['minimap2', '-ax', 'map-hifi', '--split-prefix=tmp', '-t', str(nthreads), '-o', mg_sam_out,
                       o_join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1
                       ]
        else:  # if the fastq is interleaved
            logging.info('Raw reads in interleaved file...\n')
            mem_cmd = ['minimap2', '-ax', 'sr', '--split-prefix=tmp', '-t', str(nthreads), '-o', mg_sam_out,
                       o_join(subcontig_path, mg_id + '.subcontigs.fasta'), pe1
                       ]

        if sam_size <= 0:
            logging.info('Running minimap2 on %s\n' % pe_id)
            with open(mg_sam_out, 'w') as sam_file:
                with open(o_join(abr_path, pe_id + '.stderr.txt'), 'w') as stderr_file:
                    with open(o_join(abr_path, pe_id + '.stdout.txt'), 'w') as stdout_file:
                        run_mem = Popen(mem_cmd, stdout=stdout_file, stderr=stderr_file)
                        run_mem.communicate()
        else:
            logging.info('SAM file already exists, skipping alignment...')
    else:
        logging.info('Raw FASTQ file(s) are not where you said they were...')
        sys.exit()  # TODO: replace this quick-fix with a real exception

    return pe_id, mg_sam_out


def runSamTools(abr_path, pe_id, nthreads, mg_id, mg_sam_out):
    mg_bam_out = o_join(abr_path, pe_id + '.bam')
    if isfile(mg_bam_out) == False:
        logging.info('Converting SAM to BAM with SamTools\n')
        bam_cmd = ['samtools', 'view', '-S', '-b', '-@', str(nthreads), mg_sam_out]
        with open(mg_bam_out, 'w') as bam_file:
            with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
                run_bam = Popen(bam_cmd, stdout=bam_file, stderr=stderr_file)
                run_bam.communicate()
    # sort bam file
    mg_sort_out = o_join(abr_path, pe_id + '.sorted.bam')
    if isfile(mg_sort_out) == False:
        logging.info('Sort BAM with SamTools\n')
        sort_cmd = ['samtools', 'sort', '-@', str(nthreads), mg_bam_out, '-o', mg_sort_out]
        with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
            run_sort = Popen(sort_cmd, stderr=stderr_file)
            run_sort.communicate()

    return mg_sort_out


def runMBAcov(abr_path, mg_id, sorted_bam_list):
    # run mba on sorted bams
    mg_mba_out = o_join(abr_path, mg_id + '.mbacov.tsv')
    mg_mba_std = o_join(abr_path, mg_id + '.coverage.scaled.tsv')
    try:  # if file exists but is empty
        mba_size = getsize(mg_mba_std)
    except:  # if file doesn't exist
        mba_size = -1
    if mba_size <= 0:
        logging.info('Calculating Coverage with jgi_summarize_bam_contig_depths\n')
        mba_cmd = ['jgi_summarize_bam_contig_depths', '--outputDepth', mg_mba_out]
        mba_cmd.extend(sorted_bam_list)
        with open(o_join(abr_path, mg_id + '.stderr.txt'), 'w') as stderr_file:
            run_mba = Popen(mba_cmd, stderr=stderr_file)
            run_mba.communicate()

        mg_mba_df = pd.read_csv(mg_mba_out, header=0, sep='\t')
        mg_mba_df.rename(columns={'contigName': 'subcontig_id'}, inplace=True)
        mg_mba_df.drop(columns=['contigLen', 'totalAvgDepth'], inplace=True)
        mg_mba_df.set_index('subcontig_id', inplace=True)
        scale = StandardScaler().fit(mg_mba_df.values)
        scaled_data = scale.transform(mg_mba_df.values)
        std_merge_df = pd.DataFrame(scaled_data, index=mg_mba_df.index).reset_index()
        std_merge_df.to_csv(mg_mba_std, header=True, sep='\t', index=False)

    return mg_mba_std, mg_mba_out
