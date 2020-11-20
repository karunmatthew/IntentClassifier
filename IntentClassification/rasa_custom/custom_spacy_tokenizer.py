import typing
from typing import Text, List, Any, Type

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.components import Component
from rasa_custom.custom_spacy_nlp import CustomSpacyNLP
from rasa.nlu.training_data import Message
from rasa.nlu.constants import SPACY_DOCS
from util.apputil import LANG_VISUAL_DELIMITER

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc  # pytype: disable=import-error


POS_TAG_KEY = "pos"


class CustomSpacyTokenizer(Tokenizer):
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [CustomSpacyNLP]

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
    }

    def get_doc(self, message: Message, attribute: Text) -> "Doc":
        return message.get(SPACY_DOCS[attribute])

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:

        # extract the visual information from the text information
        if LANG_VISUAL_DELIMITER in message.text:
            visual_data_string = message.text[message.text.index(LANG_VISUAL_DELIMITER) + 1 + len(LANG_VISUAL_DELIMITER):]
            message.set('visual_info', visual_data_string.strip())
            message.text = message.text[0: message.text.index(LANG_VISUAL_DELIMITER)]

        doc = self.get_doc(message, attribute)

        tokens = []
        for t in doc:
            # do not send the visual information to the feedforward layers
            # that lead to the transformer module
            if LANG_VISUAL_DELIMITER in t.text:
                break
            tokens.append(Token(t.text, t.idx, lemma=t.lemma_, data={POS_TAG_KEY: self._tag_of_token(t)}))

        return tokens

    @staticmethod
    def _tag_of_token(token: Any) -> Text:
        import spacy

        if spacy.about.__version__ > "2" and token._.has("tag"):
            return token._.get("tag")
        else:
            return token.tag_
