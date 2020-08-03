import spacy

INPUT_FILE = "outfile_pickup_simple1"
READ = "r"
OUTPUT_FILE = "outfile_pickup_simple_train"
WRITE = "w"


# load the spacy english model
nlp = spacy.load("en_core_web_lg")

file = open(INPUT_FILE, READ)
outfile = open(OUTPUT_FILE, WRITE)

actions = {'pick': 'GRASP',
           'move': 'MOVE_FORWARD',
           'take': 'GRASP',
           'go': 'MOVE_FORWARD',
           'place': 'RELEASE',
           'put': 'RELEASE',
           'carry': 'TRANSPORT'
           }

# walk

complex_actions = {
    'take to': 'TRANSPORT',
    'turn left': 'MOVE_LEFT',
    'move left': 'MOVE_LEFT',
    'move right': 'MOVE_RIGHT',
    'turn right': 'MOVE_RIGHT',
    'turn around': 'MOVE_BACK',
    'turn on': 'NOT_SUPPORTED'
}

all_actions = {}
all_actions.update(actions)
all_actions.update(complex_actions)


# returns more details for a particular identified intent
def get_intent_details(doc, verb, verb_index):
    adverb = ''
    adposition = ''
    for token in doc:
        if token.dep_ == 'advmod' and token.head.text == verb and token.head.i == verb_index and adverb == '':
            adverb = token.text
        elif token.pos_ == 'ADP' and token.head.text == verb and token.head.i == verb_index and adposition == '':
            adposition = token.text
        elif token.pos_ == 'ADV' and token.head.text == verb and token.head.i == verb_index and adverb == '':
            adverb = token.text

    return adverb, adposition


# pick the best intend from the list of possible intends
def get_intent_class(verb, adverb, adposition):
    verb = verb.strip().lower()
    verb_with_adverb = verb + ' ' + adverb.strip().lower()
    verb_with_adposition = verb + ' ' + adposition.strip().lower()

    if verb_with_adverb in complex_actions:
        return complex_actions[verb_with_adverb]
    elif verb_with_adposition in complex_actions:
        return complex_actions[verb_with_adposition]
    elif verb in actions:
        return actions[verb]
    else:
        return get_most_similar_action(verb)


# get spacy similarity score between words
def get_spacy_similarity_score(word1, word2):
    doc1 = nlp(word1)
    doc2 = nlp(word2)
    return doc1.similarity(doc2)


# returns the similarity score and the action that the extracted
# verb and modifiers are most similar to
def get_most_similar_action(verb):
    most_similar_action = 'NOT_SUPPORTED'
    most_similar_action_score = -1
    for action in all_actions.keys():
        score = get_spacy_similarity_score(action, verb)
        if score > most_similar_action_score:
            most_similar_action = action
            most_similar_action_score = score

    print("SIM SCORE :::: ", most_similar_action_score, '  ', most_similar_action)
    if most_similar_action_score > 0.6:
        return all_actions[most_similar_action]
    else:
        return 'NOT_SUPPORTED'


def generate_training_set():
    count = 0
    for line in file:
        count += 1
        if count > 2000:
            break
        line = line.strip()
        classify_intent_from_command(line)


def classify_intent_from_command(line):
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

    main_intent = ''
    for root_verb, root_verb_index in zip(root_verbs, root_verb_indices):
        root_adverb, root_adposition = get_intent_details(doc, root_verb, root_verb_index)
        if main_intent == '':
            main_intent = get_intent_class(root_verb, root_adverb, root_adposition)

    if len(root_verbs) > 1:
        print('Compound sentence')
    if len(verbs) > 0:
        for verb, verb_index in zip(verbs, verb_indices):
            adverb, adposition = get_intent_details(doc, verb, verb_index)
            other_intent = get_intent_class(verb, adverb, adposition)

            if len(root_verbs) == 0 and main_intent == '':
                # if the root word was identified as a noun
                main_intent = other_intent

            print('OTHER INTENT --- ', verb, ' ', adverb, ' ', adposition)
            print('OTHER INTENT CLASS --- ', other_intent)

    outfile.write(line + "\t" + main_intent + '\n')


generate_training_set()
file.close()
outfile.close()