# bert training
export BERT_BASE_DIR=
export DATA_DIR=
export OUTPUT_DIR=
export MAX_LENGTH=
export BATCH_SIZE=
export LEARNING_RATE=
export EPOCH=
export TASK_NAME=
 
python3 /home/mist/BERT/run_classifier.py \
--task_name=$TASK_NAME \
--do_train=true \
--do_eval=true  \
--do_predict=false \
--data_dir=$DATA_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=$MAX_LENGTH \
--train_batch_size=$BATCH_SIZE \
--learning_rate=$LEARNING_RATE \
--num_train_epochs=$EPOCH \
--output_dir=$OUTPUT_DIR