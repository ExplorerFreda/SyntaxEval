import argparse
import logging
import torch

from data import SyntheticEvaluator, NonsensicalEvaluator
from models import GPT2LM, MaskedLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm-name', type=str, default='gpt2')
    parser.add_argument('--log-path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--correction', action='store_true', default=False)
    parser.add_argument('--load-evaluator-path', type=str, default=None)
    args = parser.parse_args()

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
        model = GPT2LM(args.lm_name)
    else:
        model = MaskedLM(args.lm_name)
        assert args.batch_size == 1  # lazy approach, avoiding attention masks for pad tokens
    if args.load_evaluator_path is not None:
        evaluator_synthetic, evaluator_nonsensical = torch.load(args.load_evaluator_path)
        evaluator_nonsensical.batch_size = args.batch_size
        evaluator_synthetic.batch_size = args.batch_size
    else:
        evaluator_nonsensical = NonsensicalEvaluator(
            logger=logger, batch_size=args.batch_size, correct_sent=args.correction
        )
        evaluator_synthetic = SyntheticEvaluator(
            logger=logger, batch_size=args.batch_size, correct_sent=args.correction
        )
    if 'gpt2' in args.lm_name:
        evaluator_nonsensical.evaluate(model)
        evaluator_synthetic.evaluate(model)
    else:
        evaluator_nonsensical.evaluate_masked(model)
        evaluator_synthetic.evaluate_masked(model)
