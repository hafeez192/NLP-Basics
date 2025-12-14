'''if we try to use text.split() to tokenize a text there happens to be many problems
such as:
We love Pakistan, because it's beautiful! becomes
['We', 'love', 'Pakistan,', 'because', "it's", 'beautiful!']
, Pakistan, becomes a token
It's not handles
'''

# NOTE: solution NLTK handles these issues with good tokenizers
'''
word_tokenize()
-Removes punctuation from words

-Handles contractions

-Handles sentences cleanly

-Follows Penn Treebank rules

RegexpTokenizer()
-Custom regex tokenization

TweetTokenizer()

TreebankWordTokenizer()
'''

# Problem code 
text = "We love Pakistan, because it's beautiful!    \nAnd we really love it"
print("split based tokens")
whitespace_tokens = text.split()
print(whitespace_tokens)
# ['We', 'love', 'Pakistan,', 'because', "it's", 'beautiful!', 'And', 'we', 'really', 'love', 'it']

import nltk
'''
Why can’t word_tokenize work without punkt?

Because word_tokenize is a wrapper around:

PunktSentenceTokenizer (needs punkt)

TreebankWordTokenizer (no download required)
'''
nltk.download('punkt') #tokenizer models

from nltk.tokenize import word_tokenize, TreebankWordTokenizer
print('nltk word tokenizer')
nltk_tokens = word_tokenize(text)
print(nltk_tokens)
# ['We', 'love', 'Pakistan', ',', 'because', 'it', "'s", 'beautiful', '!', 'And', 'we', 'really', 'love', 'it']

print('TreebankTokenizer')
tbt = TreebankWordTokenizer()
treebank_tokens = tbt.tokenize(text)
print(treebank_tokens)
# ['We', 'love', 'Pakistan', ',', 'because', 'it', "'s", 'beautiful', '!', 'And', 'we', 'really', 'love', 'it']