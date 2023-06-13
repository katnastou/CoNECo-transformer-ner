[![DOI](https://zenodo.org/badge/604544128.svg)](https://zenodo.org/badge/latestdoi/604544128)
# S1000-transformer-ner
S1000 NER training for transformer models

Code for paper: S1000: A better taxonomic name corpus for biomedical information extraction

## Environment setup:
This code is tested with Python 3.9 installed with conda and the packages from requirements.txt installed in that environment. Running setup.sh will download the S1000 dataset in CoNLL format and pretrained transformer model and install the needed packages. There are some packages (spacy, scispacy) defined in requirements.txt that are not needed for running the training, but are used with the accompanying repo meant for tagging documents with the trained model https://github.com/jouniluoma/S1000-transformer-tagger

Quickstart
```
conda create -n s1000-env python=3.9
conda activate s1000-env
pip install -r requirements.txt
./setup.sh
./scripts/run-ner.sh
```
These create enviroment, installs required packages, runs training on hyperparameters set in run-ner.sh and saves the trained model.
