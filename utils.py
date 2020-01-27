import spacy
import torch 
import regex


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
    Perform sentence correction: recover sentences in natural language as much as possible 
        from a list of words. 
    Input: list[str] words: list of words before correction.
    Output: str new_sentence: sentence after correction. 
    Explanation: the reason for doing sentence correction is that the pretrained 
        models are trained on real sentences (usually with capital initial character 
        and punctuations; however, the synthetic syntax evaluation dataset provides 
        a list of words, which are all lowercased -- we might want to recover the 
        'natural' sentences from such lists of words to see the 'real' performance 
        of pretrained models. 
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
    
    # check empty sequence. 
    if len(new_sentence) == 0:
        return new_sentence
    
    # Capitalize the initial character.
    # There might be a new_sentence as follows: "we went there yesterday", he said. 
    #   - We would like to capitalize the "w" in "we", instead of the first character, i.e., left quote mark. 
    # There can also be a sentence as follows: 10 dollars were paid. 
    #   - We would not like to capitalize the "d" in "dollars". 
    # Given above, an empirical solution is to find the first English character in the sentence, which is in the
    #   first word of the sentence (i.e., not preceded by any space), and capitalize it (if applicable) to 
    #   perform sentence ocrrection. 
    # Note that the code only works for English sentences. 
    # (TODO) Consider other languages. 
    # (TODO) Handle more possible special cases. 
    new_sentence = regex.sub(
        '[ \pL]', lambda obj: obj.group(0).upper(), new_sentence, count=1)  # \pL: match letters
    return new_sentence
