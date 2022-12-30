This software are Convolutional Neural Network(cnn)-based miRNA classifier. We only use the nucleotide/base sequence of pre-miRNA to predict the classification. Based on the mechanism of generation, miRNA can be classified into mirtron and classical miRNA.

We trained our models on the partiton of human pre-miRNAs dataset and the performance was evaluated on the test dataset.

Using the well-trained model, the pre-miRNA sequence can be easily classified.

This implementation is written in python3 and numpy and tensorflow is required.

Python Requirements

Python 3.5.2

pip install numpy 
pip install tensorflow

USAGE: python isMirtron.py -s pre-miRNA sequnece
    or python isMirtron.py --sequence  pre-miRNA sequnece
for example: python isMirtron.py -s GTAAGTCTGGGGAGATGGGGGGAGCTCTGCTGAGGGTGCACAAGGCCCTGGCTCTACACACATCCCTGTCTTACAG

USAGE HELP: python isMirtron.py -h or python isMirtron --help

# Using with CSV input

In this FORK the classifier was adapted to accept CSV entries in the format:

```txt
id,seq
seq1,ACGTACGTACGTACGTACGTACGT
seq2,CGTACGTACGTACGTACGTACGTA
seq3,GTACGTACGTACGTACGTACGTAC
```

## Install

You need to create a new conda environment

```bash
cd CNNMIRTRONPRED_DIR
conda create -p .condaenv -y python=3.5
conda install -p .condaenv -c anaconda -y numpy
conda install -p .condaenv -c anaconda -y tensorflow-gpu
```

## Run - Method n 01

1. You need to activate the environment

```bash
cd CNNMIRTRONPRED_DIR
conda activate .condaenv
```

2. Classify your CSV with

```bash
python isMirtronCsv.py --csv data/samples.csv
```

## Run - Method n 02

Run directly in the environment.

```bash
cd CNNMIRTRONPRED_DIR
conda run -p .condaenv python isMirtronCsv.py --csv data/samples.csv
```

## Utility: Convert your FASTA file to CSV

```bash
echo 'id,seq' > data/samples.csv
awk -f fasta2csv.awk data/samples.fa >> data/samples.csv
```

## Utility: Convert sequence list to CSV

```bash
echo 'id,seq' > data/samples.csv
awk 'BEGIN{OFS=","} { print "seq" NR, $0 }' data/samples.txt >> data/samples.csv
```
