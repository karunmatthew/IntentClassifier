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

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        # Ensuring all the pre-requisites are met
        # We would be extracting the object and action sequence information
        # out from the message text in the tokenizer and retrieving it back here
        return [Tokenizer]

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        for example in training_data.training_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self.extract_environment_features(example, attribute)

    def process(self, message: Message, **kwargs: Any) -> None:
        self.extract_environment_features(message)

    def extract_environment_features(self, message: Message, attribute: Text = TEXT) -> None:

        sequence_features = []
        print(message.get(TOKENS_NAMES[attribute]))

        if not message.get(TOKENS_NAMES[attribute]):
            # nothing to featurize
            return

        for token in message.get(TOKENS_NAMES[attribute]):
            sequence_features.append([1, 2, 3, 4, 5, 6, 7, 8, 9])

        sentence_features = [1, 2, 3, 4, 5, 6, 7, 8]
        features = np.array(sequence_features)
        # features = np.concatenate([sequence_features, sentence_features])

        features = self._combine_with_existing_dense_features(
            message, features, DENSE_FEATURE_NAMES[attribute]
        )
        message.set(DENSE_FEATURE_NAMES[attribute], features)