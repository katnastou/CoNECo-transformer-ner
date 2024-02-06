#!/bin/bash

model="RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf"
data_dir="data"
max_seq_length="128"
batch_size="4"
learning_rate="3e-5"
epochs="1"

train_file="$data_dir/train.tsv" 
test_file="$data_dir/dev.tsv" 

echo "data dir: $data_dir" 
echo "train file: $train_file"
echo "test file : $test_file"

python3 ner_hf_trainer.py \
    --learning_rate $learning_rate \
    --num_train_epochs $epochs \
    --max_seq_length $max_seq_length \
    --batch_size $batch_size \
    --train_data "$train_file" \
    --test_data "$test_file" \
    --model_name "$model" \
    --ner_model_dir "ner-models/coneco" \
    --cache_dir "transformers-models" \
    --output_file "output" \
    --predict_position 0 \

echo "Ready."
