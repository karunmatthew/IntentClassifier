# IntentClassifier

This project uses RASA's DIET classifier for intent classification

The model that was trained with RASA's DIET was compared against a MultiLayer Perceptron.

The classifer uses both language and visual data to classify the intents behind a spoken command.
The visual data is first captured by an object detection system and converted into numerical visual features.
We use spacy's pretrained language model to generate features for the natural langugae text.

The project also includes a couple of custom components that extended the capabilities of the DIET classifier
