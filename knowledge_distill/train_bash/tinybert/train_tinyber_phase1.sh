#phase1: transfer knowledge except the predict layer
export TMP_TINYBERT_DIR=/home/mist/tinybert_imdb_phase1
export TASK_DIR=/home/mist/imdb
export TASK_NAME=imdb
export FT_BERT_BASE_DIR=/home/mist/imdb_bert_model/bert_imdb
export GENERAL_TINYBERT_DIR=/home/mist/TinyBERT_BASE
export MAX_LENGTH=
export LEARNING_RATE=
export BATCH_SIZE=
export EPOCH=

python /home/mist/TinyBERT/task_distill.py \
--teacher_model=${FT_BERT_BASE_DIR} \
--student_model=${GENERAL_TINYBERT_DIR} \
--data_dir=${TASK_DIR} \
--task_name=${TASK_NAME} \
--output_dir=${TMP_TINYBERT_DIR} \
--max_seq_length=${MAX_LENGTH} \
--train_batch_size=${BATCH_SIZE} \
--num_train_epochs=${EPOCH} \
--aug_train \
--do_lower_case  