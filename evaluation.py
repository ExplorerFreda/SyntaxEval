import argparse
import logging
import numpy as np
import os
import pickle


class Evaluator(object):
    """
        Initialize the evaluator.
        Input: 
            data_path: a folder consists of pickle files provided by Marvin and Linzen (2018).
            log_path: (default None) if given, output the result to the log path. 
    """
    def __init__(self, data_path='./LM_syneval/EMNLP2018/templates/', log_path=None, batch_size=128):

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

        if log_path is not None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            handler = logging.FileHandler(self.log_path, 'w')
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False
        else:
            self.logger = None
        
        self.batch_size = batch_size

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
