import spacy
import torch 


"""
    given a batch of lengths with size (B, ), generate length mask with size 
        (B, maxL). Positions within the sentence length are marked as 1, 
        otherwise 0
"""
def generate_length_masks(lengths, max_length=None):
    max_length = max(lengths)
    ids = torch.arange(max_length).unsqueeze(0).expand(len(lengths), -1)
    lengths = torch.tensor(lengths).unsqueeze(1).expand_as(ids)
    length_masks = (ids < lengths).float()
    if torch.cuda.is_available():
        length_masks = length_masks.cuda()
    return length_masks


quote_pairs = {
    ("'", "'"),
    ("`", "'"),
    ('"', '"'),
    ('``', '"'),
    ("``", "''"),
}
nlp = spacy.load('en')


def match(left, right):
    return left == right or ((left, right) in quote_pairs)


"""
    Perform sentence correction: recover sentences in natural language from 
        a list of words. 
"""
def correct_sentence(words):
    words = nlp(' '.join(words))
    new_words = list()
    quotes = list()  # maintain a stack of quotes
    delayed_attach = False  # enable right-combination the current token
    for w in words:
        attached = False
        attach = False
        if delayed_attach:  # deal with right attached puncts
            new_words[-1] += w.text
            delayed_attach = False
            attached = True
        if w.is_bracket:  # brackets
            if w.is_left_punct:
                delayed_attach = True
            else:
                assert w.is_right_punct
                attach = True
        elif w.is_currency or w.text in ['#', '@']:  # currency symbols
            delayed_attach = True
        elif w.text in ['-', '---']:
            attach = delayed_attach = True
        elif w.text in ['--']:
            attach = delayed_attach = False
        elif w.is_quote:
            if len(quotes) == 0 or not match(quotes[-1], w.text):  # w starts a quote
                quotes.append(w.text)
                delayed_attach = True
            else:  # w ends a quote
                assert match(quotes[-1], w.text)
                quotes = quotes[:-1]
                attach = True
        elif w.is_punct:
            attach = True
        elif w.text in ["'s", "'ll", "'d", "n't", "'m"]:  # suffixes
            attach = True

        # attach the current word
        if attached:
            continue
        if attach:
            new_words[-1] += w.text
        else:
            new_words.append(w.text)
        
    new_sentence = ' '.join(new_words)
    # capitalize the initial character
    if not 'A' <= new_sentence[0] <= 'Z':
        position = 0
        while (position < len(new_sentence) - 1) and (not 'a' <= new_sentence[position] <= 'z') and (new_sentence[position] != ' '):
            position += 1
        if len(new_sentence) > 0 and 'a' <= new_sentence[position] <= 'z':
            new_sentence = new_sentence[:position] + chr(ord('A') + ord(new_sentence[position]) - ord('a')) + new_sentence[position+1:]
    return new_sentence
        