# train designed lstm without distillnation
export OUTPUT_DIR=
export OUTPUT_EVAL_FILE=
export DATA_DIR=
# need locate pth file
export EMBEDDING_DIR=
export MAX_LENGTH=
export BATCH_SIZE_TRAIN=
export BATCH_SIZE_EVAL=
export EPOCH=
export EVAL_STEP=
export HIDDEN_SIZE=
export EMBEDDING_SIZE=

python3 /home/mist/TinyBERT/lstm_no_distill.py \
--data_dir=$DATA_DIR \
--max_seq_length=$MAX_LENGTH \
--embedding_size=$EMBEDDING_SIZE \
--n_hidden=$HIDDEN_SIZE \
--embedding_dir=$EMBEDDING_DIR \
--eval_step=$EVAL_STEP \
--eval_batch_size=$BATCH_SIZE_EVAL \
--train_batch_size=$BATCH_SIZE_TRAIN \
--num_train_epochs=$EPOCH \
--output_dir=$OUTPUT_DIR \
--output_eval_file=$OUTPUT_EVAL_FILE