# This file just documents the various features of
# spacy NLP. It also acts as a place to test out
# the various features of spacy


from spacy.lang.en import English
import spacy
from spacy.matcher.matcher import Matcher

nlp = English()

# provide a document here
doc = nlp("I three 4000")

# -------------------LEXICAL ATTRIBUTES-----------------------------------------
# ------------------------------------------------------------------------------

# doc is iterable
for token in doc:
    print(token.i)
    print(token.text)
    # lexical attributes of tokens
    print(token.is_alpha)
    print(token.is_punct)
    print(token.like_num)

# slice of the doc
span = doc[1:4]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


# load the small english model
nlp = spacy.load("en_core_web_lg")

doc = nlp("take a step to your left")
root_verb = ''
for token in doc:
    print(token.text, ' ', token.pos_, ' ', token.dep_, ' ', token.head.text)
    if token.dep_ == 'ROOT':
        root_verb = token.text

print('root ::', root_verb)

# list identified entities in the sentence
print('\nEntities in sentence ::')
for ent in doc.ents:
    print(ent.text, ' ', ent.label_)

# spacy explain function to define spacy notations
print(spacy.explain("ADV"))

# --------------------------------------------
# ---------  MATCHER -------------------------

from spacy.matcher import matcher

matcher = Matcher(nlp.vocab)
pattern = [{"LOWER": "samsung"}, {"LOWER": "note"}, {"IS_DIGIT": True}]
pattern1 = [{"LEMMA": "buy"}, {"POS": "DET", "OP": "?"}]  # ! ? + *
matcher.add('SAMSUNG_PATTERN', None, pattern)
matcher.add('BUY_PATTERN', None, pattern1)

doc = nlp("I bought a new samsung NOTE 9 phone")
matches = matcher(doc)
print('MATCHES IDENTIFIED')
for match_id, start, end in matches:
    print(doc[start:end].text)

print('-------------------------------------')
# --------------------------------------------
# ---------WORD SIMILARITY--------------------



from spacy.tokens import Doc, Span, Token

doc1 = nlp("put the cup from the counter")
doc2 = nlp("grab the spoon")

token1 = doc1[0]
token2 = doc2[0]

print(doc1.similarity(doc2))
print(token1.similarity(token2))


# ----- SPACY HASH REPRESENTATION FOR STRINGS--------

# Look up the hash for the word "cat"
cat_hash = nlp.vocab.strings["cat"]
print(cat_hash)