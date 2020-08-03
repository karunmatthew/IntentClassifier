import nltk
import spacy
# nltk.download('wordnet')
# nltk.download('wordnet_ic')

from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic

brown_ic = wordnet_ic.ic('ic-brown.dat')

# load the spacy english model
nlp = spacy.load("en_core_web_lg")


# returns all the synsets associated with the word
def get_synsets(word):
    return wordnet.synsets(word)


# returns the synset of the word that the word is mostly associated with
def get_most_common_synset(word):
    synsets = get_synsets(word)
    if len(synsets) == 1:
        return synsets[0]
    else:
        most_common_synset = None
        max_count = -1

        for synset in synsets:
            total_count = 0
            print('\n')
            for lemma in synset.lemmas():
                print(lemma.name(), ' ', lemma.count())
                if lemma.name().lower() == word.lower():
                    total_count += lemma.count()

            if total_count > max_count:
                max_count = total_count
                most_common_synset = synset

        print(most_common_synset)
        return most_common_synset


synset_1 = get_most_common_synset("grab")
synset_2 = get_most_common_synset("take")


print('Similarity Score :: ', synset_1.lin_similarity(synset_2, brown_ic))


def get_spacy_similarity_score(word1, word2):
    doc1 = nlp(word1)
    doc2 = nlp(word2)
    return doc1.similarity(doc2)


print('Spacy Similarity Score :: ', get_spacy_similarity_score("grab from", "take away"))
print('Spacy Similarity Score :: ', get_spacy_similarity_score("walk", "turn"))
print('Spacy Similarity Score :: ', get_spacy_similarity_score("walk", "left"))
print('Spacy Similarity Score :: ', get_spacy_similarity_score("walk", "right"))
print('Spacy Similarity Score :: ', get_spacy_similarity_score("walk to", "go"))