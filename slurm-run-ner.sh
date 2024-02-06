#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G


### MAHTI
###SBATCH -p gputest
###SBATCH -t 00:15:00
###SBATCH -p gpusmall
###SBATCH -t 06:00:00
###SBATCH --gres=gpu:a100:1

### PUHTI
#SBATCH -p gputest
#SBATCH -t 00:15:00
###SBATCH -p gpu
###SBATCH -t 02:00:00
#SBATCH --gres=gpu:v100:1

#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2001426
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

module purge
source venv/bin/activate

# rm -f logs/latest.out logs/latest.err
# ln -s "$SLURM_JOBID.out" "logs/latest.out"
# ln -s "$SLURM_JOBID.err" "logs/latest.err"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

OUTPUT_DIR="output/$SLURM_JOBID"
mkdir -p $OUTPUT_DIR

# comment if you don't want to delete output e.g. to save a model with best set of parameters
function on_exit {
   rm -rf "$OUTPUT_DIR"
   rm -f jobs/$SLURM_JOBID
}
trap on_exit EXIT

#check for all parameters
if [ "$#" -ne 8 ]; then
    echo "Usage: $0 model_dir data_dir max_seq_len batch_size learning_rate epochs ner_model_dir cache_dir"
    exit 1
fi

model="$1" #"RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf"
data_dir="$2" #"data"
max_seq_length="$3" #"128"
batch_size="$4" #"4"
learning_rate="$5" #"3e-5"
epochs="$6" #"1"
ner_model_dir="$7" #"ner-models/coneco"
cache_dir="$8" #"transformers-models" 

output_file="$SLURM_JOBID" #output

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
    --ner_model_dir "$ner_model_dir" \
    --cache_dir "$cache_dir" \
    --output_file "$output_file" \
    --predict_position 0 \

echo "END $SLURM_JOBID: $(date)"

