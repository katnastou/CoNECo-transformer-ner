#!/bin/bash

# Download huggingface compatible model for Roberta bio-lm
wget https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz
tar -xvzf RoBERTa-large-PM-M3-Voc-hf.tar.gz
rm RoBERTa-large-PM-M3-Voc-hf.tar.gz

# Download training data
wget https://zenodo.org/api/records/10623629/files/CoNECo_corpus_conll.tar.gz 
tar -xvzf CoNECo_corpus_conll.tar.gz
rm CoNECo_corpus_conll.tar.gz
mv {train,dev,test}.tsv data/

