import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import BertTokenizer, BertForMaskedLM
from transformers import XLNetTokenizer, XLNetLMHeadModel

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


class MaskedLM(object):
    def __init__(self, lm_name):
        if 'roberta' in lm_name:
            self.tokenizer = RobertaTokenizer.from_pretrained(lm_name)
            self.model = RobertaForMaskedLM.from_pretrained(lm_name)
        elif 'bert' in lm_name:
            self.tokenizer = BertTokenizer.from_pretrained(lm_name)
            self.model = BertForMaskedLM.from_pretrained(lm_name)
        else:
            self.tokenizer = XLNetTokenizer.from_pretrained(lm_name)
            self.model = XLNetLMHeadModel.from_pretrained(lm_name)
        self.lm_name = lm_name
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
            [self.tokenizer.cls_token_id] + \
            self.tokenizer.encode(sent) + \
            [self.tokenizer.sep_token_id] 
            for sent in sent_batch
        ]
        removed_ids = list()
        lengths = [len(sentence) for sentence in sentence_ids]
        max_length = max(lengths)        
        assert all(length == max_length for length in lengths)
        position = sum([i if sentence_ids[0][i] != sentence_ids[1][i] else 0 for i in range(len(sentence_ids[0]))])
        for i in range(len(sentence_ids)):
            removed_ids.append(sentence_ids[i][position])
            sentence_ids[i][position] = self.tokenizer.mask_token_id
        input_batch = torch.tensor(sentence_ids).long()
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()
        prob_scores = list()
        if type(self.model) is not XLNetLMHeadModel:
            _, lm_scores = self.model(input_batch, masked_lm_labels=input_batch)
            for i, index in enumerate(removed_ids):
                prob_scores.append(lm_scores[i][position][index].item())
        else:
            perm_mask = input_batch.new_zeros((input_batch.shape[0], input_batch.shape[1], input_batch.shape[1]), dtype=torch.float)
            perm_mask[:, :, position] = 1.0
            target_mapping = input_batch.new_zeros((input_batch.shape[0], 1, input_batch.shape[1]), dtype=torch.float) 
            target_mapping[:, 0, position] = 1.0
            outputs = self.model(input_batch, perm_mask=perm_mask, target_mapping=target_mapping)
            for i, index in enumerate(removed_ids):
                prob_scores.append(outputs[0][i][0][index].item())
        return prob_scores
