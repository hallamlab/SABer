import matplotlib
matplotlib.use('agg')
from os.path import join as o_join
from os.path import basename
import pandas as pd
from subprocess import Popen, PIPE
import saber.utilities as s_utils



def run_combine_recruits(final_path, ext_path, asm_path, check_path, mg_file, tetra_df_dict,
                         minhash_df, sag_list
                             ): # TODO: use logging instead of print

    mg_contigs_dict = s_utils.get_seqs(mg_file)
    mg_contigs = tuple([(r.name, r.seq) for r in mg_contigs_dict])

    for tetra_id in tetra_df_dict:
        tetra_df = tetra_df_dict[tetra_id]
        # TODO: Use full contigs instead of subcontigs for co-asm, reduces asm time for Minimus2? CISA?
        # TODO: check for co-asm files before running
        # build SAG id to SAG path dict
        sag2path_dict = {}
        for sag_path in sag_list:
            base = basename(sag_path)
            sag_id = base.rsplit('.', 1)[0]
            sag2path_dict[sag_id] = sag_path

        # Merge MinHash and GMM Tetra (passed first by ABR)
        mh_gmm_merge_df = minhash_df[['sag_id', 'contig_id']].merge(
            tetra_df[['sag_id', 'contig_id']], how='outer',
            on=['sag_id', 'contig_id']
        ).drop_duplicates()

        mh_gmm_merge_df.to_csv(o_join(final_path, tetra_id + '.final_recruits.tsv'), sep='\t', index=True)
        mg_contigs_df = pd.DataFrame(mg_contigs, columns=['contig_id', 'seq'])
        sag_de_df_list = []
        for sag_id in set(mh_gmm_merge_df['sag_id']):
            final_rec = o_join(final_path, sag_id + '.' + tetra_id + '.final_recruits.fasta')

            sub_merge_df = mh_gmm_merge_df.loc[mh_gmm_merge_df['sag_id'] == sag_id]
            print('[SABer]: Recruited %s contigs from entire analysis for %s' %
                  (sub_merge_df.shape[0], sag_id)
                  )
            with open(o_join(final_path, sag_id + '.' + tetra_id + '.final_recruits.fasta'), 'w') as final_out:
                mg_sub_filter_df = mg_contigs_df.loc[mg_contigs_df['contig_id'
                ].isin(sub_merge_df['contig_id'])
                ]
                final_mgsubs_list = ['\n'.join(['>' + x[0], x[1]]) for x in
                                     zip(mg_sub_filter_df['contig_id'],
                                         mg_sub_filter_df['seq']
                                         )
                                     ]
                final_out.write('\n'.join(final_mgsubs_list))
            '''
            # Combine SAG and final recruits # TODO: is this actually needed if MinHash is so good? I think not :)
            ext_SAG = o_join(ext_path, sag_id + '.extend_SAG.fasta')
            with open(ext_SAG, 'w') as cat_file:
                data = []
                with open(sag_file, 'r') as sag_in:
                    data.extend(sag_in.readlines())
                with open(o_join(final_path, sag_id + '.' + tetra_id + '.final_recruits.fasta'), 'r') as \
                        recruits_in:
                    data.extend(recruits_in.readlines())
                join_data = '\n'.join(data).replace('\n\n', '\n')
                cat_file.write(join_data)
            '''

            # Use BBTools dedupe.sh to deduplicate the extend SAG file
            dedupe_SAG = o_join(ext_path, sag_id + '.' + tetra_id + '.extended_SAG.fasta')
            dedupe_cmd = ['dedupe.sh', 'in=' + final_rec, 'out=' + dedupe_SAG,
                            'threads=8', 'minidentity=97', 'overwrite=true']
            run_dedupe = Popen(dedupe_cmd, stdout=PIPE)
            print(run_dedupe.communicate()[0].decode())
            de_header_list = []
            with open(dedupe_SAG, 'r') as de_file:
                data = de_file.readlines()
                for line in data:
                    if '>' in line:
                        de_header_list.append(line.strip('>').strip('\n'))
            de_sag_df = pd.DataFrame(de_header_list, columns=['contig_id'])
            de_sag_df['sag_id'] = sag_id
            de_sag_df['tetra_id'] = tetra_id
            sag_de_df_list.append(de_sag_df)
        sag_de_df = pd.concat(sag_de_df_list)
        sag_de_df.to_csv(o_join(ext_path, tetra_id + '.extended_SAGs.tsv'), sep='\t', index=True)
        '''
            # Use minimus2 to merge the SAG and the recruits into one assembly
            toAmos_cmd = ['/home/rmclaughlin/bin/amos-3.1.0/bin/toAmos', '-s',
                            ext_SAG, '-o', o_join(asm_path, sag_id + '.afg')
                            ]
            run_toAmos = Popen(toAmos_cmd, stdout=PIPE)
            print(run_toAmos.communicate()[0].decode())
            minimus_cmd = ['/home/rmclaughlin/bin/amos-3.1.0/bin/minimus2',
                            o_join(asm_path, sag_id),
                            '-D', 'REFCOUNT=0', '-D', 'OVERLAP=200', '-D', 'MINID=95'
                            ]
            run_minimus = Popen(minimus_cmd, stdout=PIPE)
            print(run_minimus.communicate()[0].decode())
            if isfile(o_join(asm_path, sag_id + '.fasta')):
                filenames = [o_join(asm_path, sag_id + '.fasta'), o_join(asm_path, sag_id + '.singletons.seq')]
                with open(o_join(asm_path, sag_id + '.minimus2.asm.fasta'), 'w') as outfile:
                    for fname in filenames:
                        with open(fname) as infile:
                            for line in infile:
                                outfile.write(line)
                move_cmd = ['mv', o_join(asm_path, sag_id + '.fasta'),
                            o_join(asm_path, sag_id + '.minimus2_no_singles.asm.fasta')
                            ]

            run_move = Popen(move_cmd, stdout=PIPE)
            clean_cmd = ['rm', '-r', o_join(asm_path, sag_id + '.runAmos.log'),
                            o_join(asm_path, sag_id + '.afg'),
                            o_join(asm_path, sag_id + '.OVL'),
                            o_join(asm_path, sag_id + '.singletons'),
                            o_join(asm_path, sag_id + '.singletons.seq'),
                            o_join(asm_path, sag_id + '.contig'),
                            o_join(asm_path, sag_id + '.ovl'),
                            o_join(asm_path, sag_id + '.coords'),
                            o_join(asm_path, sag_id + '.qry.seq'),
                            o_join(asm_path, sag_id + '.delta'),
                            o_join(asm_path, sag_id + '.bnk'),
                            o_join(asm_path, sag_id + '.ref.seq')
                            ]
            run_clean = Popen(clean_cmd, stdout=PIPE)
        '''

    # Run CheckM on all new rebuilt/updated SAGs
    print('[SABer]: Checking all new SAG quality using CheckM')
    checkm_cmd = ['checkm', 'lineage_wf', '--tab_table', '-x',
                    'fasta', '--threads', '8', '--pplacer_threads', '8', '-f',
                    o_join(check_path, 'checkM_stdout.tsv'), ext_path, check_path
                    ]
    run_checkm = Popen(checkm_cmd, stdout=PIPE)
    print(run_checkm.communicate()[0].decode())









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
        #                    if 'whole:' in x
        #                    ])*0.75))
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
        move_cmd = ['mv', cisa_outfile,    join(asm_path, sag_id + '.CISA.asm.fasta')]
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
'''
