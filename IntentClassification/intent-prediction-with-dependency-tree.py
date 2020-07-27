import spacy

INPUT_FILE = "outfile"
READ = "r"

# load the spacy english model
nlp = spacy.load("en_core_web_lg")

file = open(INPUT_FILE, READ)

action = {'pick': 'GRASP',
          'take': 'GRASP',
          'go': 'MOVE',
          'put': 'RELEASE',
          'turn': 'MOVE'}


# returns more details for a particular identified intent
def get_intent_details(doc, verb):
    adverb = ''
    adposition = ''
    for token in doc:
        if token.dep_ == 'advmod' and token.head.text == verb:
            adverb = token.text
        if token.pos_ == 'ADP' and token.head.text == verb:
            adposition = token.text

    return adverb, adposition


def classify_intent():
    count = 0
    for line in file:
        count += 1
        if count > 100:
            break
        line = line.strip()
        doc = nlp(line)
        root_verbs = []
        verbs = []
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                root_verbs.append(token.text)
            elif token.pos_ == 'VERB':
                verbs.append(token.text)

        print(line)

        for root_verb in root_verbs:
            root_adverb, root_adposition = get_intent_details(doc, root_verb)
            print('MAIN INTENT --- ', root_verb, ' ', root_adverb, ' ', root_adposition)

        if len(root_verbs) > 1:
            print('Compound sentence')

        if len(verbs) > 0:
            for verb in verbs:
                adverb, adposition = get_intent_details(doc, verb)
                print('VERB --- ', verb, ' ', adverb, ' ', adposition)
        print('\n')


classify_intent()
file.close()
