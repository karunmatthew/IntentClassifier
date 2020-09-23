from typing import Any, Dict, List, Text
import sys
from rasa.nlu.constants import DENSE_FEATURE_NAMES, TEXT
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.training_data import Message
import rasa.utils.train_utils as train_utils
import tensorflow as tf


class CustomRasaTokenizer(WhitespaceTokenizer):

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "+",
        # Text will be tokenized with case sensitive as default
        "case_sensitive": True,
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the WhitespaceTokenizer framework."""
        print("LOADING CUSTOM RASA TOKENIZER")
        super().__init__(component_config)
        # Load models for ConveRT
        model_url = "http://models.poly-ai.com/convert/v1/model.tar.gz"
        self.module = train_utils.load_tf_hub_model(model_url)

        self.tokenize_signature = self.module.signatures["tokenize"]

    def _tokenize(self, sentence: Text) -> Any:

        return self.tokenize_signature(tf.convert_to_tensor([sentence]))[
            "default"
        ].numpy()

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenize the text using the ConveRT model.

        ConveRT adds a special char in front of (some) words and splits words into
        sub-words. To ensure the entity start and end values matches the token values,
        tokenize the text first using the whitespace tokenizer. If individual tokens
        are split up into multiple tokens, we make sure that the start end end value
        of the first and last respective tokens stay the same.
        """

        # extract the visual information from the text information
        visual_data = []
        if '@@@@@@' in message.text:
            visual_data_string = message.text[message.text.index('@@@@@@')+7:]
            message.set('visual_info', visual_data_string)
            message.text = message.text[0: message.text.index('@@@@@@')]

        # perform whitespace tokenization
        tokens_in = super().tokenize(message, attribute)

        tokens_out = []
        for token in tokens_in:
            token_start, token_end, token_text = token.start, token.end, token.text

            # use ConveRT model to tokenize the text
            split_token_strings = self._tokenize(token_text)[0]

            # clean tokens (remove special chars and empty tokens)
            split_token_strings = self._clean_tokens(split_token_strings)

            tokens_out += train_utils.align_tokens(
                split_token_strings, token_end, token_start
            )

        return tokens_out

    def _clean_tokens(self, tokens: List[bytes]):
        """Encode tokens and remove special char added by ConveRT."""

        tokens = [string.decode("utf-8").replace("﹏", "") for string in tokens]
        return [string for string in tokens if string]