""" Bi-LSTM train """
from Bi_LSTM.modeling import BiLSTM
from ft_distill import DataProcessor, InputExample, InputFeatures, convert_examples_to_features, get_tensor_data
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformer.tokenization import BertTokenizer
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
import argparse
import torch
import logging
import sys
import os
import numpy as np
import pickle
from sklearn.metrics import f1_score

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()


class MyProcessor(DataProcessor):
    """Processor for my task-news classification """

    def __init__(self):
        self.labels = list(map(str, range(0, 2)))

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'train_data.csv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'dev_data.csv')), 'val')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'test_data.csv')), 'test')

    def get_labels(self):
        return self.labels

    def _create_examples(self, lines, set_type):
        """create examples for the training and val sets"""
        examples = []
        for (i, line) in enumerate(lines):
            guid = '%s-%s' % (set_type, i)
            # print("line[0]:",line[0])
            # print("line[1]:",line[1])
            # print(line)
            text_a = line[1].strip('"')
            label = line[0].strip('"')
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    
def soft_cross_entropy(predicts, targets):
    likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()

def do_eval(model, eval_dataloader, device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids_lstm, input_ids_bert, input_mask, segment_ids, label_ids, seq_lengths_lstm, seq_lengths_bert = batch_
            _, _, logits, _ = model(input_ids_lstm, seq_lengths_lstm.detach().cpu(), device)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = {"acc":  (preds == eval_labels.numpy()).mean()}
    result['f1'] = f1_score(eval_labels.numpy(), preds)
    result['eval_loss'] = eval_loss

    return result

    
def training_bilstm(args):
    """ training our designed Bi-LSTM on training dataset  """
    
    # prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # data load
    processor = MyProcessor()
    output_mode = 'classification'
    label_list = processor.get_labels()
    # lstm_tokenizer = BiLstmTokenizer(vocab_file=args.embedding_dir)
    with open('/home/mist/from_bert/bert_tokenizer.pkl', 'rb') as fi:
        bert_tokenizer = pickle.load(fi)
    train_examples = processor.get_train_examples(args.data_dir)
    train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, bert_tokenizer, output_mode)
    train_data, _ = get_tensor_data(output_mode, train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                       batch_size=args.train_batch_size)
    
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, bert_tokenizer, output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    # create model
    model = BiLSTM(embedding_size=args.embedding_size, n_hidden=args.n_hidden, n_class=len(label_list), max_len=args.max_seq_length, n_layers=1, cache_dir=args.embedding_dir)
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    size = 0
    for n, p in model.named_parameters():
        logger.info('n: {}'.format(n))
        if p.requires_grad:
            size += p.nelement()

    logger.info('Total parameters: {}'.format(size))
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    # optimizer = torch.optim.Adadelta(optimizer_grouped_parameters, lr=3e-5, rho=0.9, eps=1e-06, weight_decay=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_dev_acc = 0.
    for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0.

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
            batch = tuple(t.to(device) for t in batch)
            input_ids_lstm, input_ids_bert, input_mask, segment_ids, label_ids, seq_lengths_lstm, seq_lengths_bert = batch
            _, _, logits, _= model(input_ids_lstm, seq_lengths_lstm.detach().cpu(), device)
            loss_ce = CrossEntropyLoss()
            loss = loss_ce(logits.view(-1, len(label_list)), label_ids.view(-1))
            optimizer.zero_grad()
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()

            if (step + 1) % args.eval_step == 0:
                logger.info("***** Running evaluation *****")
                logger.info("  Epoch = {} iter {} step".format(epoch_, step))
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                model.eval()

                loss = tr_loss / (step + 1)

                result = do_eval(model, eval_dataloader, device, output_mode, eval_labels, len(label_list))
                result['global_step'] = step
                result['loss'] = loss

                result_to_file(result, args.output_eval_file)

                save_model = False

                if result['acc'] > best_dev_acc:
                    best_dev_acc = result['acc']
                    print(best_dev_acc)
                    save_model = True

                if save_model:
                    logger.info("***** Save model *****")
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_name = "pytorch_model.bin"
                    output_model_file = os.path.join(args.output_dir, model_name)
                    torch.save(model_to_save, output_model_file)


                model.train()

def test_bilstm(args):
    """ test our designed Bi-LSTM on training dataset  """
    # prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # data load
    processor = MyProcessor()
    output_mode = 'classification'
    label_list = processor.get_labels()
    # lstm_tokenizer = BiLstmTokenizer(vocab_file=args.embedding_dir)
    with open('/home/mist/from_bert/bert_tokenizer.pkl', 'rb') as fi:
        bert_tokenizer = pickle.load(fi)
    
    model_name = "pytorch_model.bin"
    output_model_file = os.path.join(args.output_dir, model_name)
    model = torch.load(output_model_file)
    
    test_examples = processor.get_test_examples(args.data_dir)
    test_features = convert_examples_to_features (test_examples, label_list, args.max_seq_length, bert_tokenizer, output_mode)
    test_data, test_labels = get_tensor_data(output_mode, test_features)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
    result = do_eval(model, test_dataloader, device, output_mode, test_labels, len(label_list))
    print(result['acc'])
    print(result['f1'])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--max_seq_length",
                        type=int,
                        default=128,
                        required=True,
                        help="The Max sequence length need for model")
    parser.add_argument("--embedding_size",
                        type=int,
                        default=300,
                        required=True,
                        help="The embedding size for input")
    parser.add_argument("--n_hidden",
                        type=int,
                        default=128,
                        required=True,
                        help="The size for hidden layer")
    parser.add_argument("--embedding_dir",
                        type=str,
                        default=None,
                        required=True,
                        help="The GloVe embedding dir.")
    parser.add_argument("--eval_step",
                        type=int,
                        default=None,
                        required=True,
                        help="every n steps check eval loss.")
    parser.add_argument("--eval_batch_size",
                        type=int,
                        default=32,
                        required=True,
                        help="batch size for eval.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=4,
                        required=True,
                        help="the num of epoch for training.")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=32,
                        required=True,
                        help="batch size for training.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        required=True,
                        help="The model save dir.")
    parser.add_argument("--output_eval_file",
                        type=str,
                        default=None,
                        required=True,
                        help="The output eval file.")
    args = parser.parse_args()
    logger.info('The args: {}'.format(args))
   
    # training_bilstm(args)
    test_bilstm(args)