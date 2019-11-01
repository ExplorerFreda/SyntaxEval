import argparse
import logging
import torch

from data import SyntheticEvaluator, NonsensicalEvaluator
from models import GPT2LM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm-name', type=str, default='gpt2')
    parser.add_argument('--log-path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--correction', action='store_true', default=False)
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

    model = GPT2LM(args.lm_name)
    evaluator_nonsensical = NonsensicalEvaluator(
        logger=logger, batch_size=args.batch_size, correct_sent=args.correction
    )
    evaluator_nonsensical.evaluate(model)
    evaluator_synthetic = SyntheticEvaluator(
        logger=logger, batch_size=args.batch_size, correct_sent=args.correction
    )
    evaluator_synthetic.evaluate(model)
