# CoNECo-transformer-ner
CoNECo NER training for transformer models

## Environment setup:
This code is tested with Python 3.9 installed with conda and the packages from requirements.txt installed in that environment. Running setup.sh will download the CoNECo dataset in CoNLL format and pretrained transformer model and install the needed packages. There are some packages (spacy, scispacy) defined in requirements.txt that are not needed for running the training, but are used with the accompanying repo meant for tagging documents with the trained model 

Quickstart
```
conda create -n venv python=3.9
conda activate venv
pip install -r requirements.txt
./setup.sh
./scripts/run-ner.sh
```
These create enviroment, installs required packages, runs training on hyperparameters set in run-ner.sh and saves the trained model.

On Puhti/Mahti supercomputers
```
module load python-data/3.9
python3.9 -m venv venv
source venv/bin/activate
python3.9 -m pip install --upgrade pip
python3.9 -m pip install --upgrade setuptools
python3.9 -m pip install -r requirements.txt
./setup.sh
sbatch slurm-run-ner.sh RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf data 128 4 3e-5 1 ner-models/coneco transformers-models
```
