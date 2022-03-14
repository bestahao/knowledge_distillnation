# running bert on test_data and generate results in output_dir
export BERT_BASE_DIR=
export DATA_DIR=
export TRAINED_CLASSIFIER=
export MAX_LENGTH=
export OUTPUT_DIR=
export TASK_NAME=
 
python3 /home/mist/BERT/run_classifier.py \
  --task_name=$TASK_NAME \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=$MAX_LENGTH \
  --output_dir=$OUTPUT_DIR