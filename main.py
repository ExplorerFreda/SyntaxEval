import argparse
import logging
import torch

from data import SyntheticEvaluator, NonsensicalEvaluator
from models import GPT2LM, MaskedLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm-name', type=str, default='gpt2',
                        help='name of language models; default: gpt2 (normal);'
                        'masked LM choices: bert-{base/large}-uncased, xlnet-{base/large}-cased, roberta-{base/large}')
    parser.add_argument('--log-path', type=str, default=None,
                        help='path to extra log file; default: no extra log file.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for batched evaluation of normal language models; default: 32.')
    parser.add_argument('--correction', action='store_true', default=False,
                        help='whether performing sentence correction (capitalizing initial char and adding punct);'
                        'default: no correction.')
    parser.add_argument('--load-evaluator-path', type=str, default=None,
                        help='path to preprocessed evaluator; default: no preprocessed evaluator.')
    args = parser.parse_args()

    # create logging path if applicable 
    if args.log_path is not None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        handler = logging.FileHandler(args.log_path, 'w')
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    else:
        logger = None

    if 'gpt2' in args.lm_name:
        model = GPT2LM(args.lm_name)  # normal language model
    else:
        model = MaskedLM(args.lm_name)  # masked language model
        assert args.batch_size == 1  # lazy approach, avoiding attention masks for pad tokens

    # load preprocessed evaluator if applicable, o.w. create evaluator from scratch
    if args.load_evaluator_path is not None:
        evaluator_synthetic, evaluator_nonsensical = torch.load(args.load_evaluator_path)
        evaluator_nonsensical.batch_size = args.batch_size
        evaluator_synthetic.batch_size = args.batch_size
    else:
        if 'gpt2' not in args.lm_name:
            raise Exception('Error: Must load preprocessed evaluator for masked language models')
        evaluator_nonsensical = NonsensicalEvaluator(
            logger=logger, batch_size=args.batch_size, correct_sent=args.correction
        )
        evaluator_synthetic = SyntheticEvaluator(
            logger=logger, batch_size=args.batch_size, correct_sent=args.correction
        )
    
    # evaluate language model 
    if 'gpt2' in args.lm_name:
        evaluator_nonsensical.evaluate(model)
        evaluator_synthetic.evaluate(model)
    else:
        evaluator_nonsensical.evaluate_masked(model)
        evaluator_synthetic.evaluate_masked(model)
