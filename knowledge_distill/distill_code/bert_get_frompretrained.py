# coding=utf-8
# Copyright 2020 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""get bert embedding layer + tokenizer."""

import random
import sys
import os
import unicodedata
import re
import logging
import csv
import argparse
import pickle


import torch
import numpy as np

from transformer import BertTokenizer, BertForMaskedLM

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_bert_model", default=None, type=str, required=True,
                        help="Downloaded pretrained model (bert-base-cased/uncased) is under this folder")
    parser.add_argument("--save_or_load", default='save', type=str, required=True,
                        help="save or load")
    args = parser.parse_args()
    # logger.info(args)

    if args.save_or_load == 'save':
        # load bert
        model = BertForMaskedLM.from_pretrained(args.pretrained_bert_model).bert
        bert_tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model, do_lower_case=True)
        embedding_layer = model.embeddings.word_embeddings

        # check model size
        size = 0
        for n, p in model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()
        print('bert', size)

        # save
        torch.save(embedding_layer.state_dict(), '/home/mist/from_bert/bert_embedding_layer.pth')
        with open('/home/mist/from_bert/bert_tokenizer.pkl', 'wb') as fid:
            pickle.dump(bert_tokenizer, fid)
    else:
        layer = torch.nn.Embedding(30522, 768)
        layer.load_state_dict(torch.load(f'/home/mist/from_bert/bert_embedding_layer.pth'))
        # check model size
        size = 0
        for n, p in layer.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()
        print(size)
        
        
    



if __name__ == "__main__":
    main()
