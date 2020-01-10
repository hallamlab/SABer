import matplotlib
matplotlib.use('agg')
from os.path import join as o_join
from os.path import basename
import pandas as pd
from subprocess import Popen, PIPE


def run_combine_recruits(final_path, ext_path, asm_path, mg_contigs, gmm_df, minhash_df, mg_subcontigs, sag_list,
                         gmm_per_pass=0.01
                         ):
    # TODO: Use full contigs instead of subcontigs for co-asm, reduces asm time for Minimus2? CISA?
    # TODO: check for co-asm files before running
    # build SAG id to SAG path dict
    sag2path_dict = {}
    for sag_path in sag_list:
        base = basename(sag_path)
        sag_id = base.rsplit('.', 1)[0]
        sag2path_dict[sag_id] = sag_path

    mg_id, mg_headers, mg_subs = mg_subcontigs
    # Count # of subcontigs recruited to each SAG
    gmm_cnt_df = gmm_df.groupby(['sag_id', 'contig_id']).count().reset_index()
    gmm_cnt_df.columns = ['sag_id', 'contig_id', 'subcontig_recruits']
    # Build subcontig count for each MG contig
    mg_contig_list = [x.rsplit('_', 1)[0] for x in mg_headers]
    mg_tot_df = pd.DataFrame(zip(mg_contig_list, mg_headers),
                             columns=['contig_id', 'subcontig_id'])
    mg_tot_cnt_df = mg_tot_df.groupby(['contig_id']).count().reset_index()
    mg_tot_cnt_df.columns = ['contig_id', 'subcontig_total']
    mg_recruit_df = gmm_cnt_df.merge(mg_tot_cnt_df, how='left', on='contig_id')
    mg_recruit_df['percent_recruited'] = mg_recruit_df['subcontig_recruits'] / \
                                         mg_recruit_df['subcontig_total']
    mg_recruit_df.sort_values(by='percent_recruited', ascending=False, inplace=True)
    # Only pass contigs that have the magjority of subcontigs recruited (>= N%)
    mg_recruit_filter_df = mg_recruit_df.loc[mg_recruit_df['percent_recruited'] >= gmm_per_pass]
    mg_contig_per_max_df = mg_recruit_filter_df.groupby(['contig_id'])[
        'percent_recruited'].max().reset_index()
    mg_contig_per_max_df.columns = ['contig_id', 'percent_max']
    mg_recruit_max_df = mg_recruit_filter_df.merge(mg_contig_per_max_df, how='left',
                                                   on='contig_id')
    # Now pass contigs that have the maximum recruit % of subcontigs
    mg_max_only_df = mg_recruit_max_df.loc[mg_recruit_max_df['percent_recruited'] >=
                                           mg_recruit_max_df['percent_max']
                                           ]

    # Merge MinHash and GMM Tetra (passed first by ABR)
    mh_gmm_merge_df = minhash_df[['sag_id', 'contig_id']].merge(
        mg_max_only_df[['sag_id', 'contig_id']], how='outer',  # gmm_df
        on=['sag_id', 'contig_id']
    ).drop_duplicates()

    mh_gmm_merge_df.to_csv(o_join(final_path, 'final_recruits.tsv'), sep='\t', index=True)

    mg_contigs_df = pd.DataFrame(mg_contigs, columns=['contig_id', 'seq'])
    for sag_id in set(mh_gmm_merge_df['sag_id']):
        sag_file = sag2path_dict[sag_id]
        sub_merge_df = mh_gmm_merge_df.loc[mh_gmm_merge_df['sag_id'] == sag_id]
        print('[SABer]: Recruited %s contigs from entire analysis for %s' %
              (sub_merge_df.shape[0], sag_id)
              )
        with open(o_join(final_path, sag_id + '.final_recruits.fasta'), 'w') as final_out:
            mg_sub_filter_df = mg_contigs_df.loc[mg_contigs_df['contig_id'
            ].isin(sub_merge_df['contig_id'])
            ]
            final_mgsubs_list = ['\n'.join(['>' + x[0], x[1]]) for x in
                                 zip(mg_sub_filter_df['contig_id'],
                                     mg_sub_filter_df['seq']
                                     )
                                 ]
            final_out.write('\n'.join(final_mgsubs_list))
        # Combine SAG and final recruits # TODO: is this actually needed if MinHash is so good? I think not :)
        with open(o_join(ext_path, sag_id + '.extend_SAG.fasta'), 'w') as cat_file:
            data = []
            with open(sag_file, 'r') as sag_in:
                data.extend(sag_in.readlines())
            with open(o_join(final_path, sag_id + '.final_recruits.fasta'), 'r') as \
                    recruits_in:
                data.extend(recruits_in.readlines())
            join_data = '\n'.join(data).replace('\n\n', '\n')
            cat_file.write(join_data)
        # TODO: Maybe create a SABer SAG for each sample? Seems like the best way to go.
        #  OR since the ASM is merged from all samples, maybe just cat all samples and
        #  assemble with SABer SAG OR original SAG as ref? Would probably use SPAdes for this.
        '''
        # Use SPAdes to co-assemble mSAG and recruits
        # TODO: use SAG as "trusted contigs" and assemble the raw reads recruited from sample(s).
        #  Get this function from Phylo(whatever) from Lulu 16S extraction from metaG's.
        print('[SABer]: Re-assembling SAG with final recruits using SPAdes')
        spades_cmd = ['/home/rmclaughlin/bin/SPAdes-3.13.0-Linux/bin/spades.py',
                      '--sc', '-k', '21,33,55,77,99,127', '--careful', '--only-assembler',
                      '-o', o_join(asm_path, sag_id), '--trusted-contigs',
                      sag_file,
                      '--s1', o_join(final_path, sag_id + '.final_recruits.fasta')
                      ]
        run_spades = Popen(spades_cmd, stdout=PIPE)
        print(run_spades.communicate()[0].decode())
        move_cmd = ['mv', o_join(asm_path, sag_id + 'scaffolds.fasta'),
                    o_join(asm_path, sag_id + '.SPAdes.asm.fasta')
                    ]

        run_move = Popen(move_cmd, stdout=PIPE)
        print(run_move.communicate()[0].decode())
        clean_cmd = ['rm', '-rf', o_join(asm_path, sag_id)]
        run_clean = Popen(clean_cmd, stdout=PIPE)
        print(run_clean.communicate()[0].decode())
        '''

    '''
        # Use CISA to integrate the SAG and Recruited contigs
        asm_sag_path = join(asm_path, sag_id)
        if not path.exists(asm_sag_path):
            makedirs(asm_sag_path)
        print('[SABer]: Building Merge config file')
        merge_config = join(asm_sag_path, sag_id + '_merge_config')
        with open(merge_config, 'w') as merge_out:
            count = '2'
            final_recruits_path = join(final_path, sag_id + '.final_recruits.fasta')
            min_len = '100'
            master_file = join(asm_sag_path, sag_id + '.merged.ctg.fasta')
            gap = '11'
            config_list = ['count='+count, 'data='+sag_file+',title=SAG',
                            'data='+final_recruits_path+',title=final_recruits',
                            'min_length='+min_len, 'Master_file='+master_file,
                            'Gap='+gap
                            ]
            merge_out.write('\n'.join(config_list))
        print('[SABer]: Merging SAG with final recruits')
        merge_cmd = ['python2.7', '/home/rmclaughlin/bin/CISA1.3/Merge.py', merge_config]
        run_merge = Popen(merge_cmd, stdout=PIPE)
        merge_stdout = run_merge.communicate()[0].decode()
        #genome_len = str(int(sum([int(x.split(':')[1]) for x in str(merge_stdout).split('\n')
        #					if 'whole:' in x
        #					])*0.75))
        genome_len = str([int(x.split(':')[1]) for x in str(merge_stdout).split('\n')
                            if 'whole:' in x
                            ][0])
    
        print('[SABer]: Building CISA config file')
        cisa_config = join(asm_sag_path, sag_id + '_cisa_config')
        with open(cisa_config, 'w') as cisa_out:
            cisa_outfile = join(asm_sag_path, sag_id + '.CISA.ctg.fasta')
            nucmer_path = '/home/rmclaughlin/bin/MUMmer3.23/nucmer'
            r2_gap = '0.95'
            cisa_path = '/home/rmclaughlin/bin/CISA1.3'
            makeblastdb_path = '/home/rmclaughlin/anaconda3/bin/makeblastdb'
            blastn_path = '/home/rmclaughlin/anaconda3/bin/blastn'
    
            cisa_config_list = ['genome='+genome_len, 'infile='+master_file,
                            'outfile='+cisa_outfile, 'nucmer='+nucmer_path,
                            'R2_Gap='+r2_gap, 'CISA='+cisa_path,
                            'makeblastdb='+makeblastdb_path, 'blastn='+blastn_path,
                            'workpath='+asm_sag_path 
                            ]
            cisa_out.write('\n'.join(cisa_config_list))
        print('[SABer]: Integrating SAG with final recruits using CISA')
    
        cisa_cmd = ['python2.7', '/home/rmclaughlin/bin/CISA1.3/CISA.py', cisa_config]
        run_cisa = Popen(cisa_cmd, stdout=PIPE, cwd=asm_sag_path)
        cisa_stdout = run_cisa.communicate()[0].decode()
        print(cisa_stdout)
        move_cmd = ['mv', cisa_outfile,	join(asm_path, sag_id + '.CISA.asm.fasta')]						
        run_move = Popen(move_cmd, stdout=PIPE)
        clean_cmd = ['rm', '-rf', asm_sag_path]
        run_clean = Popen(clean_cmd, stdout=PIPE)
    
        # Use SPAdes to co-assemble mSAG and recruits
        # TODO: use SAG as "trusted contigs" and assemble the raw reads recruited from sample(s)
        print('[SABer]: Re-assembling SAG with final recruits using SPAdes')
        spades_cmd = ['/home/rmclaughlin/bin/SPAdes-3.13.0-Linux/bin/spades.py',
                        '--sc', '-k', '21,33,55,77,99,127', '--careful', '--only-assembler',
                        '-o', join(asm_path, sag_id), '--trusted-contigs',
                        sag_file,
                        '--s1', join(final_path, sag_id + '.final_recruits.fasta')
                        ]
        run_spades = Popen(spades_cmd, stdout=PIPE)
        print(run_spades.communicate()[0].decode())
        move_cmd = ['mv', join(join(asm_path, sag_id),'scaffolds.fasta'),
                        join(asm_path, sag_id + '.SPAdes.asm.fasta')
                        ]
    
        run_move = Popen(move_cmd, stdout=PIPE)
        print(run_move.communicate()[0].decode())
        clean_cmd = ['rm', '-rf', join(asm_path, sag_id)]
        run_clean = Popen(clean_cmd, stdout=PIPE)
        print(run_clean.communicate()[0].decode())
    
        # Use minimus2 to merge the SAG and the recruits into one assembly
        toAmos_cmd = ['/home/rmclaughlin/bin/amos-3.1.0/bin/toAmos', '-s',
                        join(ext_path, sag_id + '.extend_SAG.fasta'), '-o',
                        join(asm_path, sag_id + '.afg')
                        ]
        run_toAmos = Popen(toAmos_cmd, stdout=PIPE)
        print(run_toAmos.communicate()[0].decode())
        minimus_cmd = ['/home/rmclaughlin/bin/amos-3.1.0/bin/minimus2',
                        join(asm_path, sag_id),
                        '-D', 'REFCOUNT=0', '-D', 'OVERLAP=200', '-D', 'MINID=95'
                        ]
        run_minimus = Popen(minimus_cmd, stdout=PIPE)
        print(run_minimus.communicate()[0].decode())
        if isfile(join(asm_path, sag_id + '.fasta')):
            filenames = [join(asm_path, sag_id + '.fasta'), join(asm_path, sag_id + '.singletons.seq')]
            with open(join(asm_path, sag_id + '.minimus2.asm.fasta'), 'w') as outfile:
                for fname in filenames:
                    with open(fname) as infile:
                        for line in infile:
                            outfile.write(line)
            move_cmd = ['mv', join(asm_path, sag_id + '.fasta'),
                        join(asm_path, sag_id + '.minimus2_no_singles.asm.fasta')
                        ]
    
        run_move = Popen(move_cmd, stdout=PIPE)
        clean_cmd = ['rm', '-r', join(asm_path, sag_id + '.runAmos.log'),
                        join(asm_path, sag_id + '.afg'),
                        join(asm_path, sag_id + '.OVL'),
                        join(asm_path, sag_id + '.singletons'),
                        join(asm_path, sag_id + '.singletons.seq'),
                        join(asm_path, sag_id + '.contig'),
                        join(asm_path, sag_id + '.ovl'),
                        join(asm_path, sag_id + '.coords'),
                        join(asm_path, sag_id + '.qry.seq'),
                        join(asm_path, sag_id + '.delta'),
                        join(asm_path, sag_id + '.bnk'),
                        join(asm_path, sag_id + '.ref.seq')
                        ]
        run_clean = Popen(clean_cmd, stdout=PIPE)
    
    # Run CheckM on all new rebuilt/updated SAGs
    print('[SABer]: Checking all new SAG quality using CheckM')
    checkm_cmd = ['checkm', 'lineage_wf', '--tab_table', '-x',
                    'fasta', '--threads', '8', '--pplacer_threads', '8', '-f',
                    join(check_path, 'checkM_stdout.tsv'), asm_path, check_path
                    ]
    run_checkm = Popen(checkm_cmd, stdout=PIPE)
    print(run_checkm.communicate()[0].decode())
    '''