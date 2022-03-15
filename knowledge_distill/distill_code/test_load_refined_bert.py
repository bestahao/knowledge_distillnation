from transformer.modeling import TinyBertForSequenceClassification 

def main():
    # teacher_model = TinyBertForSequenceClassification.from_pretrained('/home/mist/sst_bert_model/bert_sst', from_tf=True, from_tf_modify=True, num_labels=2)
    student_mode = TinyBertForSequenceClassification.from_pretrained('/home/mist/TinyBERT_BASE', num_labels=2)
    
if __name__ == '__main__':
    main()
    