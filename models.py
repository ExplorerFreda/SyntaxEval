import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import generate_length_masks


class GPT2LM(object):
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