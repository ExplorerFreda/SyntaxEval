import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from evaluation import Evaluator


def generate_length_masks(lengths, max_length=None):
    max_length = max(lengths)
    ids = torch.arange(max_length).unsqueeze(0).expand(len(lengths), -1)
    lengths = torch.tensor(lengths).unsqueeze(1).expand_as(ids)
    length_masks = (ids < lengths).float()
    if torch.cuda.is_available():
        length_masks = length_masks.cuda()
    return length_masks


class GPTLMEval(object):
    def __init__(self, lm_name):
        self.tokenizer = GPT2Tokenizer.from_pretrained(lm_name)
        self.model = GPT2LMHeadModel.from_pretrained(lm_name)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    """
        Input: 
            sent_batch: a list of (untokenized) sentences. 
        Output:
            prob_scores: a list of float, indicating probability scores
            of the input sentences. 
    """
    def prob_score(self, sent_batch):
        sentence_ids = [
            [self.tokenizer.bos_token_id] + \
            self.tokenizer.encode(sent) + \
            [self.tokenizer.eos_token_id] 
            for sent in sent_batch
        ]
        lengths = [len(sentence) for sentence in sentence_ids]
        max_length = max(lengths)
        padded_sent_ids = [
            item + [self.tokenizer.eos_token_id] * (max_length - lengths[i])
            for i, item in enumerate(sentence_ids)
        ]
        input_batch = torch.tensor(padded_sent_ids).long()
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()
        _, lm_scores, _ = self.model(input_batch, labels=input_batch)
        log_prob = self.collect_log_prob(input_batch, lm_scores, lengths)
        return log_prob.tolist()

    @staticmethod
    def collect_log_prob(id_batch, lm_scores, lengths):
        # generate length masks
        lm_log_probs = lm_scores.log_softmax(dim=-1)[:, :-1].contiguous().view(
            -1, lm_scores.shape[-1])
        length_masks = generate_length_masks(lengths)[:, 1:]
        labels = id_batch[:, 1:].contiguous().view(-1)
        selected_log_probs = lm_log_probs[torch.arange(labels.shape[0]), labels]
        selected_log_probs = selected_log_probs.contiguous().view(
            id_batch.size(0), -1)
        valid_log_probs = selected_log_probs * length_masks
        return valid_log_probs.sum(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm-name', type=str, default='gpt2')
    parser.add_argument('--log-path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    model = GPTLMEval(args.lm_name)
    evaluator = Evaluator(
        correct_sent=True, log_path=args.log_path, batch_size=args.batch_size
    )
    evaluator.evaluate(model)
