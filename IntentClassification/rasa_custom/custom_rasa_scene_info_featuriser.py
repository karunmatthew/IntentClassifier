import numpy as np
from typing import Any, Optional, Text, List, Type

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    TEXT,
    LANGUAGE_MODEL_DOCS,
    DENSE_FEATURE_NAMES,
    DENSE_FEATURIZABLE_ATTRIBUTES,
    SEQUENCE_FEATURES,
    SENTENCE_FEATURES, TOKENS_NAMES,
)


class SceneInfoFeaturiser(DenseFeaturizer):
    """Featurizer using transformer based language models.

    Uses the output of HFTransformersNLP component to set the sequence and sentence
    level representations for dense featurizable attributes of each message object.
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        print('SCENE INFO')
        return [Tokenizer]

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        for example in training_data.training_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self._set_lm_features(example, attribute)


    def process(self, message: Message, **kwargs: Any) -> None:
        """Sets the dense features from the language model doc to the incoming
        message."""
        self._set_lm_features(message)

    def _set_lm_features(self, message: Message, attribute: Text = TEXT) -> None:

        sequence_features = []

        print(message.get(TOKENS_NAMES[attribute]))

        if not message.get(TOKENS_NAMES[attribute]):
            # nothing to featurize
            return

        for token in message.get(TOKENS_NAMES[attribute]):
            sequence_features.append([1, 2, 3, 4, 5, 6, 7, 8])

        sentence_features = [1, 2, 3, 4, 5, 6, 7, 8]
        features = np.array(sequence_features)
        # features = np.concatenate([sequence_features, sentence_features])

        features = self._combine_with_existing_dense_features(
            message, features, DENSE_FEATURE_NAMES[attribute]
        )
        message.set(DENSE_FEATURE_NAMES[attribute], features)