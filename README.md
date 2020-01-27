# A Toolkit for Fast Morphosyntactic Evaluation
This toolkit performs fast evaluation on two morphosyntactic evaluation tasks:
- Marvin and Linzen, *"[Targeted syntactic evaluation of language models](https://www.aclweb.org/anthology/D18-1151)"*, EMNLP 2018
- Gulordava et al., *"[Colorless green recurrent networks dream hierarchically](https://www.aclweb.org/anthology/N18-1108.pdf)"*, NAACL 2018

It takes **only ~3 minutes** to evaluate a language model on all the tasks of targetd syntactic evaluation.

## Dependencies
Python 3.7 <br>
SpaCy <br>
PyTorch >= 1.2.0 <br>
transformers == 2.0.0 (for the evaluation of a pretrained language model)

## Usage

Example usage:
```
git clone --recursive https://github.com/ExplorerFreda/SyntaxEval.git
python filter_datasets.py  # preprocessing
python python main.py --lm-name gpt2 --batch-size 32 --load-evaluator-path ./data/common evaluators.pt
```

To evaluate a custom language model, you will need a language model class with a member function `prob_score`.

The `prob_score` function takes a batch of strings (i.e., untokenized sentences) as input, and outputs a list of float indicating the "probability score" of each sentence according to the language model. 

### Definition: Probability Scores
Let *P(s; LM)* denote the probability of sentence *s* given by the language model *LM*. Then a function *f(s)* is a valid probability score function using this language model, if and only if the following statement is true:

For any two sentences *s1* and *s2*, if *P(s1; LM) >= P(s2; LM)*, then *f(s1) >= f(s2)*. 

For example, log probability is a valid probability score function. 

Note: the definition above implies that same sentences must have same probability score. 

## Example
The following command evaluates GPT-2 model:
```bash
python main.py 
```
For detailed or customized evaluation, please refer to the arguments in `main.py`. 


