import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import sys
import numpy as np 
import gzip

def json_to_dataframe(json_file_path):
    if json_file_path.rsplit('.', 1)[1] == 'gz':
        with gzip.open(json_file_path, 'rt', encoding='utf-8') as file:
            data = json.load(file)
    else:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    df = pd.DataFrame.from_dict(data, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Step'}, inplace=True)
    return df

def nested_json_to_dataframe(json_file_path):
    if json_file_path.rsplit('.', 1)[1] == 'gz':
        with gzip.open(json_file_path, 'rt', encoding='utf-8') as file:
            data = json.load(file)
    else:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    processed_data = []
    for step, details in data.items():
        entry = {"Step": step}
        for key, value in details.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    entry[f"{key}_{subkey}"] = subvalue
            else:
                entry[key] = value
        processed_data.append(entry)
    return pd.DataFrame(processed_data)

def parse_out(file_path):
    with open(file_path, 'r') as file:
            data = file.read()
            count_list = [x for x in data.split('[CoverM]') 
                         if (('reads mapped out of' in x) and
                            ('_1.qc.decon.filtered.fq.gz')
                        in x)][0].split(' ')
            found_ind = count_list.index('found')
            found_count = int(count_list[found_ind + 1])
            total_ind = count_list.index('total')
            total_count = int(count_list[total_ind - 1])
            return total_count, found_count

def process_sample(sample_dir):
    # Define paths for the expected files
    file_paths = {
        "conmap": os.path.join(sample_dir, "read_mapping/results/contig_stats.json.gz"),
        "magmap": os.path.join(sample_dir, "binning/results/mag_stats.json"),
        "magqc": os.path.join(sample_dir, "mag_qc/results/mag_qc_stats.json"),
        "magtax": os.path.join(sample_dir,"mag_profiling/results/mag_profiling.ncbi_majority_vote.tsv")
    }

    # Check if all required files exist
    if not all(os.path.exists(path) for path in file_paths.values()):
        return None  # Skip this sample if any file is missing

    # Process each file and extract necessary counts
    conmap_df = json_to_dataframe(file_paths["conmap"])
    magmap_df = nested_json_to_dataframe(file_paths["magmap"])
    magqc_df = nested_json_to_dataframe(file_paths["magqc"])
    magtax_df = pd.read_csv(file_paths["magtax"], sep='\t', header=0)
    mag_qc_read_df = pd.merge(magqc_df, magmap_df, on='Step')
    mag_qc_read_df = pd.merge(mag_qc_read_df, magtax_df, left_on='Step', right_on='Genome ID')
    mag_qc_read_df['Ralstonia'] = [True if 'Ralstonia' in x else False for x in mag_qc_read_df['GTDB classification']]
    mag_qc_read_df['sample_id'] = sample_dir.split('/')[1]
    return mag_qc_read_df

def extract_taxonomy_level(taxonomy, level_index):
    parts = taxonomy.split(';')
    if len(parts) > level_index and parts[level_index]:
        return parts[level_index].split('__')[1]
    return 'Unknown'

def create_contamination_completeness_plot(mag_file, output_path):
    # Load the data
    mag_data = pd.read_csv(mag_file, sep='\t', header=0)
    group_data = mag_data[['MAG_ID', 'sample_id', 'Genome ID', 'GTDB classification', 'completeness', 'contamination', 'coverage', 'quality', '16S_rRNA', 'ASV_PAIR']]
    group_data['ASV_PAIR'] = group_data['ASV_PAIR'].astype(str)
    group_data = group_data.sort_values(by='ASV_PAIR', ascending=False)
    group_data = group_data.drop_duplicates(subset='MAG_ID', keep='first')

    # filter out Ralstonia
    group_data['Ralstonia'] = [True if 'Ralstonia' in x else False  for x in group_data['GTDB classification']]
    group_data = group_data[group_data['Ralstonia'] == False]
    group_data['log(RPM)'] = np.log(group_data['coverage'])

    # Extract phylum (level 1) and class (level 2) from MAG_Taxonomy
    group_data['Phylum'] = group_data['GTDB classification'].apply(lambda x: extract_taxonomy_level(x, 1))
    group_data['Class'] = group_data['GTDB classification'].apply(lambda x: extract_taxonomy_level(x, 2))

    # Filter data based on specified criteria
    filtered_data = group_data#[(group_data['completeness'] >= 50) & (group_data['contamination'] <= 10)]

    # Create a scatterplot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=filtered_data,
        x='completeness',
        y='contamination',
        hue='quality',
        hue_order=['low', 'medium', 'high'],
        palette={'high': 'black', 'medium': 'grey', 'low': 'white'},
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

# Add the dashed red lines for the top and left borders of the region
    # Left side line (Vertical line from (90, 0) to (90, 5))
    plt.vlines(
        x=90, 
        ymin=-5, 
        ymax=5, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Top side line (Horizontal line from (90, 5) to (100, 5))
    plt.hlines(
        y=5, 
        xmin=90, 
        xmax=105, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Add the dashed red lines for the top and left borders of the region
    # Left side line (Vertical line from (90, 0) to (90, 5))
    plt.vlines(
        x=50, 
        ymin=-5, 
        ymax=10, 
        colors='grey', 
        linestyles='dashed', 
        linewidth=1
    )

    # Top side line (Horizontal line from (90, 5) to (100, 5))
    plt.hlines(
        y=10, 
        xmin=50, 
        xmax=105, 
        colors='grey', 
        linestyles='dashed', 
        linewidth=1
    )

    # Set the x and y axis limits
    plt.xlim(-0.5, 101)
    plt.ylim(-0.5, 17)

    # Adjust plot aesthetics
    plt.xlabel('Completeness (%)')
    plt.ylabel('Contamination (%)')
    plt.title('Scatter Plot of Completeness vs Contamination by DataType')

    # Adjust the legend
    leg = plt.legend(title='DataType', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)

    # Manually set edge colors for the legend markers
    for handle in leg.legendHandles:
        handle.set_edgecolor('black')
        handle.set_linewidth(0.5)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the plot
    plt.savefig(output_path + '_All.pdf', bbox_inches='tight', format='pdf')
    plt.savefig(output_path + '_All.png', bbox_inches='tight', format='png', dpi=300,)
    plt.close()


    # Filter data based on specified criteria
    filtered_data = group_data.loc[group_data['quality'].isin(['high', 'medium'])]
    filtered_data['Has_16S'] = ['True' if x != 0 else 'False' for x in filtered_data['16S_rRNA']]

    # Create a scatterplot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=filtered_data,
        x='completeness',
        y='contamination',
        size='log(RPM)',
        hue='quality',
        hue_order=['medium', 'high'],
        palette={'high': 'black', 'medium': 'grey'},
        sizes=(5, 400),
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add the dashed red lines for the top and left borders of the region
    # Left side line (Vertical line from (90, 0) to (90, 5))
    plt.vlines(
        x=90, 
        ymin=-5, 
        ymax=5, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Top side line (Horizontal line from (90, 5) to (100, 5))
    plt.hlines(
        y=5, 
        xmin=90, 
        xmax=105, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Set the x and y axis limits
    plt.xlim(49, 101)
    plt.ylim(-0.5, 10.5)

    # Adjust plot aesthetics
    plt.xlabel('Completeness (%)')
    plt.ylabel('Contamination (%)')
    plt.title('Scatter Plot of Completeness vs Contamination by DataType')

    # Adjust the legend
    leg = plt.legend(title='DataType', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)

    # Manually set edge colors for the legend markers
    for handle in leg.legendHandles:
        handle.set_edgecolor('black')
        handle.set_linewidth(0.5)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the plot
    plt.savefig(output_path + '_MQHQ_RPM.pdf', bbox_inches='tight', format='pdf')
    plt.savefig(output_path + '_MQHQ_RPM.png', bbox_inches='tight', format='png', dpi=300,)
    plt.close()

    # Create a scatterplot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=filtered_data,
        x='completeness',
        y='contamination',
        size='log(RPM)',
        hue='Has_16S',
        hue_order=['False', 'True'],
        palette={'True': 'black', 'False': 'white'},
        sizes=(5, 400),
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add the dashed red lines for the top and left borders of the region
    # Left side line (Vertical line from (90, 0) to (90, 5))
    plt.vlines(
        x=90, 
        ymin=-5, 
        ymax=5, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Top side line (Horizontal line from (90, 5) to (100, 5))
    plt.hlines(
        y=5, 
        xmin=90, 
        xmax=105, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Set the x and y axis limits
    plt.xlim(49, 101)
    plt.ylim(-0.5, 10.5)

    # Adjust plot aesthetics
    plt.xlabel('Completeness (%)')
    plt.ylabel('Contamination (%)')
    plt.title('Scatter Plot of Completeness vs Contamination by DataType')

    # Adjust the legend
    leg = plt.legend(title='DataType', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)

    # Manually set edge colors for the legend markers
    for handle in leg.legendHandles:
        handle.set_edgecolor('black')
        handle.set_linewidth(0.5)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the plot
    plt.savefig(output_path + '_MQHQ_16S.pdf', bbox_inches='tight', format='pdf')
    plt.savefig(output_path + '_MQHQ_16S.png', bbox_inches='tight', format='png', dpi=300,)
    plt.close()

    # Create a scatterplot
    plt.figure(figsize=(10, 8))
    print(filtered_data.head())
    sns.scatterplot(
        data=filtered_data,
        x='completeness',
        y='contamination',
        size='log(RPM)',
        hue='ASV_PAIR',
        hue_order=['False', 'True'],
        palette={'True': 'black', 'False': 'white'},
        sizes=(5, 400),
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add the dashed red lines for the top and left borders of the region
    # Left side line (Vertical line from (90, 0) to (90, 5))
    plt.vlines(
        x=90, 
        ymin=-5, 
        ymax=5, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Top side line (Horizontal line from (90, 5) to (100, 5))
    plt.hlines(
        y=5, 
        xmin=90, 
        xmax=105, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Set the x and y axis limits
    plt.xlim(49, 101)
    plt.ylim(-0.5, 10.5)

    # Adjust plot aesthetics
    plt.xlabel('Completeness (%)')
    plt.ylabel('Contamination (%)')
    plt.title('Scatter Plot of Completeness vs Contamination by DataType')

    # Adjust the legend
    leg = plt.legend(title='DataType', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)

    # Manually set edge colors for the legend markers
    for handle in leg.legendHandles:
        handle.set_edgecolor('black')
        handle.set_linewidth(0.5)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the plot
    plt.savefig(output_path + '_MQHQ_ASV_PAIR.pdf', bbox_inches='tight', format='pdf')
    plt.savefig(output_path + '_MQHQ_ASV_PAIR.png', bbox_inches='tight', format='png', dpi=300,)
    plt.close()

    # Create a scatterplot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=filtered_data,
        x='completeness',
        y='contamination',
        size='log(RPM)',
        hue='Phylum',
        sizes=(5, 400),
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add the dashed red lines for the top and left borders of the region
    # Left side line (Vertical line from (90, 0) to (90, 5))
    plt.vlines(
        x=90, 
        ymin=-5, 
        ymax=5, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Top side line (Horizontal line from (90, 5) to (100, 5))
    plt.hlines(
        y=5, 
        xmin=90, 
        xmax=105, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Set the x and y axis limits
    plt.xlim(49, 101)
    plt.ylim(-0.5, 10.5)

    # Adjust plot aesthetics
    plt.xlabel('Completeness (%)')
    plt.ylabel('Contamination (%)')
    plt.title('Scatter Plot of Completeness vs Contamination by DataType')

    # Adjust the legend
    leg = plt.legend(title='DataType', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)

    # Manually set edge colors for the legend markers
    for handle in leg.legendHandles:
        handle.set_edgecolor('black')
        handle.set_linewidth(0.5)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the plot
    plt.savefig(output_path + '_MQHQ_Phylum.pdf', bbox_inches='tight', format='pdf')
    plt.savefig(output_path + '_MQHQ_Phylum.png', bbox_inches='tight', format='png', dpi=300,)
    plt.close()






if len(sys.argv) < 2:
    print("Usage: python script.py <parent_directory_containing_sample_dirs>")
    sys.exit(1)

parent_dir = sys.argv[1]
aggregated_data = []
mag_qc_read_list = []
# Iterate over each sample directory and process

sample_list = ['GMCF-1942b-I-07-9865739-9865752', 'GMCF-1942b-I-08-9865739-9865753', 'GMCF-1942b-I-09-9865739-9865754', 'GMCF-1942b-I-04-9865738-9865749',
              'GMCF-1942b-I-05-9865738-9865750', 'GMCF-1942b-I-06-9865738-9865751', 'GMCF-1942b-I-01-9865737-9865746', 'GMCF-1942b-I-02-9865737-9865747', 
              'GMCF-1942b-I-03-9865737-9865748'#, 'GMCF-1942b-I-N-NTC', 'GMCF-1942b-I-P-PC'
              ]

mag_file = 'MAG_CONCAT_DATA.tsv'
#output_path = sys.argv[2]
output_path = "MAG_CC_plot"

for sample_dir in os.listdir(parent_dir):
    for s in sample_list:
        if s in sample_dir:
            print(sample_dir)
            full_sample_path = os.path.join(parent_dir, sample_dir)
            if os.path.isdir(full_sample_path):
                mag_qc_read_df = process_sample(full_sample_path)
                mag_qc_read_list.append(mag_qc_read_df)

concat_df = pd.concat(mag_qc_read_list)
concat_df['MAG_ID'] = [x for x in concat_df['Step']]
concat_df.to_csv(mag_file, sep='\t', index=False)
genome_qc_df = pd.read_csv('./GMCF-1942b_bins/Master_genome_QC.csv', sep=',')
genome_qc_df['MAG_ID'] = [x.rsplit('.', 1)[0] for x in genome_qc_df['Genome_Id']]
asv2mag_df = pd.read_csv('../blast_results.MG_MAG_contigs.tsv', sep='\t')


merge_df = genome_qc_df.merge(concat_df, on='MAG_ID')
merge_df = merge_df.merge(asv2mag_df, on='MAG_ID', how='left')
asv_pair_list = []
for idx, row in merge_df.iterrows():
    if (float(row['pident']) >= 99.9 and int(row['length']) >= 200):
        asv_pair_list.append('True')
    else:
        asv_pair_list.append('False')
merge_df['ASV_PAIR'] = asv_pair_list    
print(merge_df.head())    
merge_df.to_csv(mag_file, sep='\t', index=False)

create_contamination_completeness_plot(mag_file, output_path)








