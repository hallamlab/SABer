#!/bin/bash
# This script is run on the ./xpg directory created by a standard SABer analysis.
# Software dependencies:
#		minimap2
#		samtools
#		sed
#		seqkit
#		spades.py
#		checkm
#
# Inputs:
#		xPG.fasta - found in SABer .xpg/ output directory
#		R1.fastq - FASTQ file of forward raw reads
#		R2.fastq - FASTQ file of reverse raw reads
#
# Intermediates:
#		[xPG]_raw.sam
#		[xPG]_raw.bam
#		[xPG]_raw.sorted.bam
#		[xPG]_raw.sorted.mapped.bam
#		[xPG]_raw.read_names.txt
#		[xPG]_raw.read_names.R1.txt
#		[xPG]_raw.read_names.R2.txt
#		[xPG]_raw.1.fq
#		[xPG]_raw.2.fq
#		NOTE: most intermeditates are deleted to save for space!
#
# Outputs
#		reasm/[xPG]/[spades output]		

xPG=$1
FQDIR=$2
OUTDIR=$3
THREADS=$4

# trim file to get base name
BASE=$(basename $xPG | rev | cut -d'.' -f2- | rev)
echo $BASE

# Make the output directory for assemblies
REASM=${OUTDIR}/reassemblies/${BASE}
mkdir -p ${REASM}

# list FASTQs in FQDIR
FQS=$(ls $FQDIR/*.1.fq.gz)
for f1 in $FQS;
	do
	# Get rev reads
	f2=$(echo ${f1/.1.fq.gz/.2.fq.gz})
	
	# trim fwd for base id
	FQBASE=$(basename $f1 | cut -d'.' -f1)
	# Align raw FASTQs to the xPG
	minimap2 -ax sr -t ${THREADS} ${xPG} ${f1} ${f2} | \
		samtools sort -o ${OUTDIR}/${BASE}.${FQBASE}.sorted.bam -
	
	samtools view -b -f 1 -F 12 -@ ${THREADS} \
               ${OUTDIR}/${BASE}.${FQBASE}.sorted.bam | \
               samtools sort -o ${OUTDIR}/${BASE}.${FQBASE}.final.bam -
	
	rm ${OUTDIR}/${BASE}.${FQBASE}.sorted.bam
	
	done;

# Merge all bams
samtools merge ${OUTDIR}/${BASE}.final.bam ${OUTDIR}/${BASE}.*.final.bam
samtools sort -n ${OUTDIR}/${BASE}.final.bam -o ${OUTDIR}/${BASE}.final_sorted.bam
rm -rf ${OUTDIR}/${BASE}.*.final.bam
rm -rf ${OUTDIR}/${BASE}.final.bam

# Output mapped FASTQs from BAMs
bamToFastq -i ${OUTDIR}/${BASE}.final_sorted.bam \
	-fq ${OUTDIR}/${BASE}.mapped.1.fq \
	-fq2 ${OUTDIR}/${BASE}.mapped.2.fq 

rm -rf ${OUTDIR}/${BASE}.final_sorted.bam

# Repair read pairing just in case
repair.sh in=${OUTDIR}/${BASE}.mapped.1.fq \
	in2=${OUTDIR}/${BASE}.mapped.2.fq \
	out=${OUTDIR}/${BASE}.mapped.1.fq.gz \
	out2=${OUTDIR}/${BASE}.mapped.2.fq.gz \
	ain=t qin=33

rm -rf ${OUTDIR}/${BASE}.mapped.*.fq

bbduk.sh in1=${OUTDIR}/${BASE}.mapped.1.fq.gz \
	in2=${OUTDIR}/${BASE}.mapped.2.fq.gz \
	out1=${OUTDIR}/${BASE}.qced.1.fq.gz \
	out2=${OUTDIR}/${BASE}.qced.2.fq.gz \
	ref=/home/mcglock/mambaforge/envs/reasm/opt/bbmap-39.01-0/resources/adapters.fa \
	ktrim=r k=23 mink=11 hdist=1 tpe tbo \
	qtrim=rl trimq=10 minlen=75

rm -rf ${OUTDIR}/${BASE}.mapped.*.fq.gz

# Re-assemble the xPG mapped raw reads with SPAdes
spades.py -o ${REASM} --isolate --trusted-contigs ${xPG} \
	--pe1-1 ${OUTDIR}/${BASE}.qced.1.fq.gz \
	--pe1-2 ${OUTDIR}/${BASE}.qced.2.fq.gz

# Move contigs and scaffolds
mv ${REASM}/contigs.fasta ${REASM}.reasm.contigs.fasta
mv ${REASM}/scaffolds.fasta ${REASM}.reasm.scaffolds.fasta

# Clean up all the intermediates
rm -rf ${OUTDIR}/${BASE}.qced.1.fq* \
	${OUTDIR}/${BASE}.qced.2.fq* \
	${REASM}
