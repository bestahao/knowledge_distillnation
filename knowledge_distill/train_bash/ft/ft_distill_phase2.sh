#phase2: train ft distill
export TASK_DIR=
export TASK_NAME=
export FT_BERT_BASE_DIR=
# need delete the folder which generate from phase1
export MODEL_DIR=
export TOKENIZER_DIR=
# need locate the pytorch_model.bin
export AUTOENCODER_DIR=
export MAX_LENGTH=
export BATCH_SIZE=
export LEARNING_RATE=
export EPOCH=
export EVAL_STEP=
export HIDDEN_in_LSTM=
export EMBEDDING_SIZE=
export HIDDEN_in_AUTOENCODER=

python /home/mist/TinyBERT/ft_distill.py \
--aug_train \
--do_lower_case \
--teacher_model=${FT_BERT_BASE_DIR} \
--bert_tokenizer=${TOKENIZER_DIR} \
--data_dir=${TASK_DIR} \
--task_name=${TASK_NAME} \
--output_dir=${MODEL_DIR} \
--learning_rate=${LEARNING_RATE}  \
--num_train_epochs=${EPOCH} \
--eval_step=${EVAL_STEP} \
--max_seq_len=${MAX_LENGTH} \
--train_batch_size=${BATCH_SIZE} \
--embedding_size=${EMBEDDING_SIZE} \
--hidden_size4lstm=${HIDDEN_in_LSTM} \
--hidden_size4autoencoder=${HIDDEN_in_AUTOENCODER} \
--autoencoder_output_dir ${AUTOENCODER_DIR} \
