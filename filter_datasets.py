import argparse
import logging
from pprint import pprint
import torch
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer

from data import SyntheticEvaluator, NonsensicalEvaluator
from models import GPT2LM


if __name__ == '__main__':
    evaluator_nonsensical = NonsensicalEvaluator(correct_sent=True)
    evaluator_synthetic = SyntheticEvaluator(correct_sent=True)

    pprint(evaluator_synthetic.group_metadata())
    pprint(evaluator_nonsensical.group_metadata())

    for tokenizer_name in ['bert-base-uncased', 'bert-large-uncased', 'xlnet-base-cased', 'xlnet-large-cased', 'roberta-base', 'roberta-large']:
        print(tokenizer_name)
        if 'roberta' in tokenizer_name:
            tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        elif 'bert' in tokenizer_name:
            tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        else:
            tokenizer = XLNetTokenizer.from_pretrained(tokenizer_name)
        evaluator_synthetic.filter_by_tokenizer(tokenizer)

    pprint(evaluator_synthetic.group_metadata())
    pprint(evaluator_nonsensical.group_metadata())
    torch.save((evaluator_synthetic, evaluator_nonsensical), './data/common_evaluators.pt')
