import spacy

INPUT_FILE = "outfile1"
READ = "r"

# load the spacy english model
nlp = spacy.load("en_core_web_lg")

file = open(INPUT_FILE, READ)

action = {'pick': 'GRASP',
          'move': 'MOVE',
          'take': 'GRASP',
          'go': 'MOVE',
          'place': 'RELEASE',
          'put': 'RELEASE',
          }
# should turn be move ??
# walk

complex_action = {
                  'take to': 'TRANSPORT',
                  'turn left': 'MOVE_LEFT',
                  'turn right': 'MOVE_RIGHT',
                  'turn around': 'MOVE_BACK',
                  'turn on': 'NOT_SUPPORTED'
                  }


# returns more details for a particular identified intent
def get_intent_details(doc, verb, verb_index):
    adverb = ''
    adposition = ''
    for token in doc:
        if token.dep_ == 'advmod' and token.head.text == verb and token.head.i == verb_index and adverb == '':
            adverb = token.text
        elif token.pos_ == 'ADP' and token.head.text == verb and token.head.i == verb_index and adposition == '':
            adposition = token.text

    return adverb, adposition


# pick the best intend from the list of possible intends
def get_intent_class(verb, adverb, adposition):
    verb = verb.strip().lower()
    verb_with_adverb = verb + ' ' + adverb.strip().lower()
    verb_with_adposition = verb + ' ' + adposition.strip().lower()

    if verb_with_adverb in complex_action:
        return complex_action[verb_with_adverb]
    elif verb_with_adposition in complex_action:
        return complex_action[verb_with_adposition]
    elif verb in action:
        return action[verb]
    else:
        return 'NOT_SUPPORTED'


def classify_intent():
    count = 0
    for line in file:
        count += 1
        if count > 200:
            break
        line = line.strip()
        classify_intent_from_command(line)


def classify_intent_from_command(data):
    print(data)
    cols = data.split('\t')
    line = cols[0]
    print(line)
    tag = cols[1]
    doc = nlp(line)
    root_verbs = []
    root_verb_indices = []
    verbs = []
    verb_indices = []

    for token in doc:
        print(token.text, ' ', token.pos_, ' ', token.dep_, ' ', token.head.text)
        if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
            root_verbs.append(token.text)
            root_verb_indices.append(token.i)
        elif token.pos_ == 'VERB':
            verbs.append(token.text)
            verb_indices.append(token.i)
    print(line)
    # print(root_verbs)
    # print(root_verb_indices)
    # print(verbs)
    # print(verb_indices)
    main_intent = ''
    for root_verb, root_verb_index in zip(root_verbs, root_verb_indices):
        root_adverb, root_adposition = get_intent_details(doc, root_verb, root_verb_index)
        print('MAIN INTENT --- ', root_verb, ' ', root_adverb, ' ', root_adposition)
        if main_intent == '':
            main_intent = get_intent_class(root_verb, root_adverb, root_adposition)
        print('MAIN INTENT CLASS --- ', main_intent)

    if len(root_verbs) > 1:
        print('Compound sentence')
    if len(verbs) > 0:
        for verb, verb_index in zip(verbs, verb_indices):
            adverb, adposition = get_intent_details(doc, verb, verb_index)
            print('OTHER INTENT --- ', verb, ' ', adverb, ' ', adposition)
            print('OTHER INTENT CLASS --- ', get_intent_class(verb, adverb, adposition))
    print("ACTUAL CLASS :: ", tag)
    if tag == main_intent:
        print('MATCH')
    else:
        print('NOT MATCH')
    print('\n')


classify_intent()
# classify_intent_from_command('take a step to your left')

file.close()
