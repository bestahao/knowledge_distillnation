#phase2: transfer knowledge only the predict layer
export TMP_TINYBERT_DIR=
export TASK_DIR=
export TASK_NAME=
export FT_BERT_BASE_DIR=
export TINYBERT_DIR=
export MAX_LENGTH=
export LEARNING_RATE=
export BATCH_SIZE=
export EPOCH=
export EVAL_STEP=
export EMBEDDING_SIZE=
export TEMPERATURE=


python /home/mist/TinyBERT/task_distill.py \
--pred_distill  \
--aug_train  \
--do_lower_case \
--teacher_model=${FT_BERT_BASE_DIR} \
--student_model=${TMP_TINYBERT_DIR} \
--data_dir=${TASK_DIR} \
--task_name=${TASK_NAME} \
--output_dir=${TINYBERT_DIR} \
--learning_rate=${LEARNING_RATE}  \
--num_train_epochs=${EPOCH}  \
--eval_step=${EVAL_STEP} \
--max_seq_length=${MAX_LENGTH} \
--train_batch_size=${BATCH_SIZE} \
--temperature=${TEMPERATURE}