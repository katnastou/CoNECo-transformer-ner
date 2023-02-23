#!/bin/bash

# Download huggingface compatible model for Roberta bio-lm
wget https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz
tar -xvzf RoBERTa-large-PM-M3-Voc-hf.tar.gz
rm RoBERTa-large-PM-M3-Voc-hf.tar.gz

# Download training data
wget https://a3s.fi/s1000/s1000-conll.tar.gz
tar -xvzf s1000-conll.tar.gz
rm s1000-conll.tar.gz


