# data augumentation
export BERT_BASE_DIR=
export GLOVE_EMB=
export GLUE_DIR=
export TASK_NAME=

python /home/mist/TinyBERT/data_augmentation.py \
--pretrained_bert_model ${BERT_BASE_DIR} \
--glove_embs ${GLOVE_EMB} \
--glue_dir ${GLUE_DIR} \
--task_name ${TASK_NAME}
