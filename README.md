# A Toolkit for Fast Targeted Syntactic Evaluation
This toolkit takes **only ~3 minutes** to evaluate a language model on all the tasks of targetd syntactic evaluation ([Marvin and Linzen, 2018](https://www.aclweb.org/anthology/D18-1151)).

## Dependencies
PyTorch 1.1.0 (I have not tested the latest PyTorch, but it should work) <br>
transformers (formerly pytorch_transformers and pytorch_pretrained_bert)

## Usage
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
python example.py 
```


## Citation
If you find this software useful in your research, please consider citing it as follows:
```
@software{shi2019toolkit,
  author = {Haoyue Shi},
  title = {A Toolkit for Fast Targeted Syntactic Evaluation},
  url = {https://github.com/explorerfreda/syntaxeval}
  year = {2019}
}
```
