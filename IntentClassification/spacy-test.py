from spacy.lang.en import English
import spacy

nlp = English()

# provide a document here
doc = nlp("I three 1990")

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
print(spacy.explain("dobj"))