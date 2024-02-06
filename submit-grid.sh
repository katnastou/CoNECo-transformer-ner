#!/bin/bash

MAX_JOBS=100

mkdir -p jobs

MODELS="
RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf
"

DATA_DIRS="
data
"
### test grid
# BATCH_SIZES="2"
# LEARNING_RATES="5e-5" 
# EPOCHS="1 2"
# REPETITIONS=1

### grid
BATCH_SIZES="2 4 8 16 32"
LEARNING_RATES="5e-5 4e-5 3e-5 2e-5 1e-5 5e-6"
EPOCHS="1 2 3 4 5 6 7 8 9 10"
REPETITIONS=3
#BATCH_SIZES="8"

#LEARNING_RATES="5e-5 3e-5 1e-5"

#EPOCHS="5"

#REPETITIONS=1
seq_len="128"
ner_model_dir="ner-models/coneco"
cache_dir="transformers-models"

for repetition in `seq $REPETITIONS`; do
	for batch_size in $BATCH_SIZES; do
	    for learning_rate in $LEARNING_RATES; do
            for epochs in $EPOCHS; do
                for model in $MODELS; do
                    for data_dir in $DATA_DIRS; do
                        while true; do
                            jobs=$(ls jobs | wc -l)
                            if [ $jobs -lt $MAX_JOBS ]; then break; fi
                            echo "Too many jobs ($jobs), sleeping ..."
                            sleep 60
                        done
                        echo "Submitting job with params $model $data_dir $seq_len $batch_size $learning_rate $epochs"
                        job_id=$(
                        sbatch slurm-run-ner.sh \
                            $model \
                            $data_dir \
                            $seq_len \
                            $batch_size \
                            $learning_rate \
                            $epochs \
                            $ner_model_dir \
                            $cache_dir \
                            | perl -pe 's/Submitted batch job //'
                        )
                        echo "Submitted batch job $job_id"
                        touch jobs/$job_id
                        sleep 10
                    done
                done
            done
	    done
	done
done
