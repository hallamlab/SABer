import argparse
import os
import subprocess
import tempfile
import shutil

def run_checkm_on_batch(batch, output_dir, checkm_args):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output_dir = os.path.join(temp_dir, "checkm_output")
        os.makedirs(temp_output_dir, exist_ok=True)

        # Prepare batch directory
        batch_dir = os.path.join(temp_dir, "batch")
        os.makedirs(batch_dir, exist_ok=True)

        for genome in batch:
            shutil.copy(genome, batch_dir)

        checkm_cmd = ["checkm", "lineage_wf", batch_dir, temp_output_dir] + checkm_args
        checkm_str = ' '.join(checkm_cmd)
        print(f'Running: {checkm_str}')
        subprocess.run(checkm_cmd, check=True)

        
        # Move results to the final output directory
        for file_name in os.listdir(temp_output_dir):
            full_file_name = os.path.join(temp_output_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, output_dir)
        
def compile_results(output_dir, final_output):
    result_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                    if ((os.path.isfile(os.path.join(output_dir, f))) & ('checkm_output' in f))
                    ]
    with open(final_output, 'w') as outfile:
        for i, fname in enumerate(result_files):
            with open(fname) as infile:
                if i != 0:
                    infile.readline()  # skip header for all files except the first
                shutil.copyfileobj(infile, outfile)

def find_index(my_list, value):
    try:
        return my_list.index(value) + 1
    except ValueError:
        return -1  # or any other value to indicate not found


def main():
    parser = argparse.ArgumentParser(description='Wrapper script to run CheckM in batches.')
    parser.add_argument('genomes_dir', type=str, help='Directory containing genome files.')
    parser.add_argument('output_dir', type=str, help='Directory to store CheckM results.')
    parser.add_argument('--batchsize', type=int, required=True, help='Number of genomes to process in each batch.')

    # Add CheckM arguments to the main parser
    parser.add_argument('-r', '--reduced_tree', action='store_true', help='Use reduced tree (requires <16GB of memory) for determining lineage of each bin')
    parser.add_argument('--ali', action='store_true', help='Generate HMMER alignment file for each bin')
    parser.add_argument('--nt', action='store_true', help='Generate nucleotide gene sequences for each bin')
    parser.add_argument('-g', '--genes', action='store_true', help='Bins contain genes as amino acids instead of nucleotide contigs')
    parser.add_argument('-u', '--unique', type=int, help='Minimum number of unique phylogenetic markers required to use lineage-specific marker set (default: 10)')
    parser.add_argument('-m', '--multi', type=int, help='Maximum number of multi-copy phylogenetic markers before defaulting to domain-level marker set (default: 10)')
    parser.add_argument('--force_domain', action='store_true', help='Use domain-level sets for all bins')
    parser.add_argument('--no_refinement', action='store_true', help='Do not perform lineage-specific marker set refinement')
    parser.add_argument('--individual_markers', action='store_true', help='Treat marker as independent (i.e., ignore co-located set structure)')
    parser.add_argument('--skip_adj_correction', action='store_true', help='Do not exclude adjacent marker genes when estimating contamination')
    parser.add_argument('--skip_pseudogene_correction', action='store_true', help='Skip identification and filtering of pseudogenes')
    parser.add_argument('--aai_strain', type=float, help='AAI threshold used to identify strain heterogeneity (default: 0.9)')
    parser.add_argument('-a', '--alignment_file', type=str, help='Produce file showing alignment of multi-copy genes and their AAI identity')
    parser.add_argument('--ignore_thresholds', action='store_true', help='Ignore model-specific score thresholds')
    parser.add_argument('-e', '--e_value', type=float, help='E-value cut off (default: 1e-10)')
    parser.add_argument('-l', '--length', type=float, help='Percent overlap between target and query (default: 0.7)')
    parser.add_argument('-f', '--file', type=str, help='Print results to file (default: stdout)')
    parser.add_argument('--tab_table', action='store_true', help='Print tab-separated values table')
    parser.add_argument('-x', '--extension', type=str, default='fna', help='Extension of bins (other files in directory are ignored) (default: fna)')
    parser.add_argument('-t', '--threads', type=int, help='Number of threads (default: 1)')
    parser.add_argument('--pplacer_threads', type=int, help='Number of threads used by pplacer (memory usage increases linearly with additional threads) (default: 1)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress console output')
    parser.add_argument('--tmpdir', type=str, help='Specify an alternative directory for temporary files')

    args = parser.parse_args()

    checkm_args = []
    for arg in vars(args):
        if arg not in ['genomes_dir', 'output_dir', 'batchsize']:
            value = getattr(args, arg)
            if value == True:
                checkm_args.append(f"--{arg}")
            elif ((value != None) & (value != False)):
                checkm_args.append(f"--{arg}")
                checkm_args.append(str(value))

    genome_files = [os.path.join(args.genomes_dir, f) for f in os.listdir(args.genomes_dir) if f.endswith(args.extension)]
    os.makedirs(args.output_dir, exist_ok=True)
    file_ind = find_index(checkm_args, '--file')
    file_temp = checkm_args[file_ind]
    
    for i, n in enumerate(range(0, len(genome_files), args.batchsize)):
        file_val = file_temp.replace('.tsv', '.' + str(i) + '.tsv')
        checkm_args[file_ind] = file_val
        batch = genome_files[n:n + args.batchsize]
        run_checkm_on_batch(batch, args.output_dir, checkm_args)
    
    # Compile all results into a single file
    final_output_file = os.path.join(args.output_dir, file_temp)
    compile_results(args.output_dir, final_output_file)
    print("All batches completed. Results compiled in:", final_output_file)

if __name__ == "__main__":
    main()

