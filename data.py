import argparse
import numpy as np
import os
import pickle
import utils
from utils import correct_sentence


class Evaluator(object):
    def __init__(self):
        pass

    """
        Returns group metadata, i.e., how many instances in each evaluation group. 
    """
    def group_metadata(self):
        metadata = dict()
        for group_name in self.data:
            metadata[group_name] = len(self.data[group_name])
        return metadata

    """
        Filter out the sentences which can be evaluated by a specific model, as BERT is a masked language model based on BPE. 
        Thus, it can only tackle syntactic tests which consists of two sentences with the same number of tokens. 
        Input:
            tokenizer: the tokenizer of a given model, defined by the huggingface repo. 
                Docs can be found here: https://huggingface.co/transformers
    """
    def filter_by_tokenizer(self, tokenizer):
        new_data = dict()
        for group_name in self.data:
            new_data[group_name] = list()
            for eval_instance in self.data[group_name]:
                tokenized_sents = [tokenizer.tokenize(sent) for sent in eval_instance]
                remove_flag = False
                for sent in tokenized_sents[1:]:
                    if len(tokenized_sents[0]) != len(sent):
                        remove_flag = True
                        break
                    different_points = sum([1 if tokenized_sents[0][i] != sent[i] else 0 for i in range(len(sent))])
                    if different_points > 1:
                        remove_flag = True
                        break
                if not remove_flag:
                    new_data[group_name].append(eval_instance)
        self.data = new_data


class SyntheticEvaluator(Evaluator):
    """
        Initialize the evaluator.
        Input: 
            data_path: a folder consists of pickle files provided by Marvin and Linzen (2018).
            logger: (default None) if given, output the result to the log path. 
            correct_sent: whether performing a sentence correction (capitalize initial character, add punctuation etc.).
    """
    def __init__(self, data_path='./LM_syneval/EMNLP2018/templates/', 
            logger=None, batch_size=128, correct_sent=False):
        super(SyntheticEvaluator, self).__init__()
        self.groups = [
            ['Simple', ['simple_agrmt']],
            ['In a sentential complement', ['sent_comp']],
            ['Short VP coordination', ['vp_coord']], 
            ['Long VP coordination', ['long_vp_coord']], 
            ['Across a prepositional phrase', ['prep_anim', 'prep_inanim']], 
            ['Across a subject relative clause', ['subj_rel']], 
            ['Across an object relative clause', ['obj_rel_across_anim', 'obj_rel_across_inanim']], 
            ['Across an object relative (no that)', ['obj_rel_no_comp_across_anim', 'obj_rel_no_comp_across_inanim']], 
            ['In an object relative clause', ['obj_rel_within_anim', 'obj_rel_within_inanim']], 
            ['In an object relative (no that)', ['obj_rel_no_comp_within_anim', 'obj_rel_no_comp_within_inanim']], 
            ['Simple ref. ana.', ['simple_reflexives']], 
            ['In a sentential complement ref. ana.', ['reflexive_sent_comp']],
            ['Across a relative clause ref. ana.', ['reflexives_across']],
            ['Simple NPI', ['simple_npi_anim', 'simple_npi_inanim']],
            ['Across a relative clause NPI', ['npi_across_anim', 'npi_across_inanim']]
        ]
        self.data = dict()
        for group in self.groups:
            group_name = group[0]
            self.data[group_name] = list()
            for fn in group[1]:
                data = pickle.load(open(os.path.join(data_path, fn + '.pickle'), 'rb'))
                for key in data:
                    self.data[group_name].extend(data[key])

        self.logger = logger
        self.batch_size = batch_size
        self.correct_sent = correct_sent

    """
        Run evaluation for a specific model. 
        Input: 
            model: a language model, which is able to calculate "probability 
            score" of a batch of sentences, using 
                model.prob_score(sent_batch)
            Here, sent_batch is a list of strings denoting (untokenized)
            sentences. 
        "Probability score" of a language model is defined as follows:
            Let P(s) denote the probability of sentence s, output by the 
            language model. For any two sentences s1 and s2, if 
            P(s1) >= P(s2), the probability score of s1 should be no less 
            than that of s2. For example, log probability is a valid 
            probability score. 
    """
    def evaluate(self, model, print_name=True):
        for group in self.groups:
            group_name = group[0]
            sentences, scores, lbs, rbs = list(), list(), list(), list()
            correct_cnt = test_cnt = 0
            for instance in self.data[group_name]:
                lbs.append(len(sentences))
                sentences.extend(instance)
                rbs.append(len(sentences))
            # TODO(freda): use correct_sentence in utils.py
            if self.correct_sent:  
                for i, sent in enumerate(sentences):
                    if 'a' <= sent[0] <= 'z':
                        sent = chr(ord(sent[0]) - ord('a') + ord('A')) + sent[1:]
                    if sent[-1] != '.':
                        sent = sent + '.'
                    sentences[i] = sent
            for start in range(0, len(sentences), self.batch_size):
                end = min(len(sentences), start + self.batch_size)
                sent_batch = sentences[start:end]
                batch_score = model.prob_score(sent_batch)
                scores.extend(batch_score)
            for i, lb in enumerate(lbs):
                rb = rbs[i]
                correct_label = 'T' if scores[lb] > max(scores[lb:rb])-1e-10 else 'F'
                correct_cnt += 1 if correct_label == 'T' else 0 
                test_cnt += 1
                if self.logger is not None:
                    self.logger.info('{:s}\t{:8.5f}\t{:s}'.format(correct_label, scores[lb], sentences[lb]))
                    for idx in range(lb+1, rb):
                        self.logger.info(' \t{:8.5f}\t{:s}'.format(scores[idx], sentences[idx]))
            if print_name:
                print(group_name, '{:.2f}'.format(float(correct_cnt)/test_cnt))
            else:
                print('{:.2f}'.format(float(correct_cnt)/test_cnt))


class NonsensicalEvaluator(Evaluator):
    """
        Initialize the evaluator.
        Input: 
            data_path: a folder consists of pickle files provided by Gulordava et al. (2018).
            log_path: (default None) if given, output the result to the log path. 
    """
    def __init__(self, data_path='./colorlessgreenRNNs/data/agreement/English', 
            logger=None, batch_size=128, correct_sent=False):
        super(NonsensicalEvaluator, self).__init__()
        self.data = {'Nonsensical': list()}
        curr_sent_pair = list()
        info_path = os.path.join(data_path, 'generated.tab')
        for i, info in enumerate(open(info_path).readlines()[1:]):
            info = info.split('\t')
            assert (i % 2 == 0) == (info[5] == 'correct')
            assert (i % 2 == 1) == (info[5] == 'wrong')
            sentence = info[-1].replace('""', '"').split()[:-1]
            if sentence[0][0] == '"':
                sentence = [sentence[0][1:]] + sentence[1:]
            len_prefix = int(info[-2])
            sentence[len_prefix] = info[4]
            curr_sent_pair.append(sentence)
            if len(curr_sent_pair) == 2:
                self.data['Nonsensical'].append(curr_sent_pair)
                curr_sent_pair = list()
        self.logger = logger
        self.batch_size = batch_size
        self.correct_sent = correct_sent

    """
        Run evaluation for a specific model. 
        Input: 
            model: a language model, which is able to calculate "probability 
            score" of a batch of sentences, using 
                model.prob_score(sent_batch)
            Here, sent_batch is a list of strings denoting (untokenized)
            sentences. 
        "Probability score" of a language model is defined as follows:
            Let P(s) denote the probability of sentence s, output by the 
            language model. For any two sentences s1 and s2, if 
            P(s1) >= P(s2), the probability score of s1 should be no less 
            than that of s2. For example, log probability is a valid 
            probability score. 
    """
    def evaluate(self, model, print_name=True):
        for group_name in self.data:
            sentences, scores, lbs, rbs = list(), list(), list(), list()
            correct_cnt = test_cnt = 0
            for instance in self.data[group_name]:
                lbs.append(len(sentences))
                sentences.extend(instance)
                rbs.append(len(sentences))
            if self.correct_sent:
                for i, sent in enumerate(sentences):
                    sentences[i] = correct_sentence(sent)
            else:
                for i, sent in enumerate(sentences):
                    sentences[i] = ' '.join(sent)
            for start in range(0, len(sentences), self.batch_size):
                end = min(len(sentences), start + self.batch_size)
                sent_batch = sentences[start:end]
                batch_score = model.prob_score(sent_batch)
                scores.extend(batch_score)
            for i, lb in enumerate(lbs):
                rb = rbs[i]
                correct_label = 'T' if scores[lb] > max(scores[lb:rb])-1e-10 else 'F'
                correct_cnt += 1 if correct_label == 'T' else 0 
                test_cnt += 1
                if self.logger is not None:
                    self.logger.info('{:s}\t{:8.5f}\t{:s}'.format(correct_label, scores[lb], sentences[lb]))
                    for idx in range(lb+1, rb):
                        self.logger.info(' \t{:8.5f}\t{:s}'.format(scores[idx], sentences[idx]))
            if print_name:
                print(group_name, '{:.2f}'.format(float(correct_cnt)/test_cnt))
            else:
                print('{:.2f}'.format(float(correct_cnt)/test_cnt))
