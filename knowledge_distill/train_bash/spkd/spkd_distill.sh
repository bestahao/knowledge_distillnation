export TASK_DIR=
export TASK_NAME=
export FT_BERT_BASE_DIR=
export MODEL_DIR=
export TOKENIZER_DIR=
export EMBEDDING_DIR=
export MAX_LENGTH=
export LEARNING_RATE=
export BATCH_SIZE=
export EPOCH=
export EVAL_STEP=
export HIDDEN_SIZE=
export EMBEDDING_SIZE=

python /home/mist/TinyBERT/spkd_distill.py \
--aug_train \
--do_lower_case \
--teacher_model ${FT_BERT_BASE_DIR} \
--bert_tokenizer ${TOKENIZER_DIR} \
--data_dir ${TASK_DIR} \
--task_name ${TASK_NAME} \
--output_dir ${MODEL_DIR} \
--learning_rate ${LEARNING_RATE}  \
--num_train_epochs ${EPOCH} \
--eval_step ${EVAL_STEP} \
--max_seq_len ${MAX_LENGTH} \
--train_batch_size ${BATCH_SIZE} \
--embedding_size ${EMBEDDING_SIZE} \
--hidden_size4lstm ${HIDDEN_SIZE}