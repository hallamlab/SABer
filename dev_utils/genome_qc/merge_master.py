import os
import pandas as pd
import warnings
import plotly.express as px
import plotly.io as pio
import sys


parent_dir = sys.argv[1]
file_1 = 'barrnap_subunit_counts.tsv'
file_2 = 'dedupe_mqhp.tsv'
file_3 = 'trnascan_trna_counts.tsv'
out_final = 'Master_genome_QC.csv'
master = pd.DataFrame()

# loop over all directories in the parent directory
out_path = os.path.join(parent_dir, out_final)

#check if the directory is a folder
first_file_path = os.path.join(parent_dir, "barrnap", file_1)
second_file_path = os.path.join(parent_dir, "dedupe", file_2)
third_file_path = os.path.join(parent_dir, "trnascan", file_3)

#read the file into a pandas dataframe
bar_df = pd.read_csv(first_file_path,sep='\t')

#create a new dataframe with the ID as index
bar_piv_df = bar_df.pivot(index='genome_id', columns='name', values='subunit_count').reset_index().fillna(0)

#Read the dedupe_dataframe 
dedupe_df = pd.read_csv(second_file_path,sep='\t')

#Merge the two dataframes
merged_df = pd.merge(dedupe_df, bar_piv_df, left_on='Bin Id', right_on='genome_id', how='left').drop('genome_id', axis=1)
merged_df.fillna(0, inplace=True)
#read the file into a pandas dataframe
trna_df = pd.read_csv(third_file_path,sep='\t')
merge2_df = pd.merge(merged_df, trna_df, left_on='Bin Id', right_on='genome_id', how='left').drop('genome_id', axis=1)

#Create the final master table
master = pd.concat([master, merge2_df])
master[['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']] = master['classification'].str.split(';', expand=True)

# remove prefix from each column
master['Domain'] = master['Domain'].str.replace('d__', '')
master['Phylum'] = master['Phylum'].str.replace('p__', '')
master['Class'] = master['Class'].str.replace('c__', '')
master['Order'] = master['Order'].str.replace('o__', '')
master['Family'] = master['Family'].str.replace('f__', '')
master['Genus'] = master['Genus'].str.replace('g__', '')
master['Species'] = master['Species'].str.replace('s__', '')

final = master.drop('classification', axis=1)

#Save the master table
final.to_csv(out_path, index=False)
if '16S_rRNA' in final.columns:
	final['contains_16S'] = [False if x == 0 else True for x in final['16S_rRNA']]
else:
	final['contains_16S'] = [False for x in final.index]

#Contamination and Completion plots
pio.templates.default = "plotly_dark"
fig = px.scatter(final, x="Completeness", y="Contamination", color="Class",
				 symbol='contains_16S', hover_data=['Phylum'],
				 title="Contamination v/s Completion plot",
				 color_discrete_sequence=px.colors.qualitative.Pastel
				 )
fig.update_traces(marker={'size': 20})
fig.update_yaxes(autorange="reversed")
fig.write_html(os.path.join(parent_dir, "CC_plot.html"))

