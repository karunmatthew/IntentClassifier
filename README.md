# IntentClassifier

This is a baseline model based on a MultiLayer Perceptron

The baseline uses both language and visual data to classify the intents behind a spoken command.
The visual data is first captured by an object detection system and converted into numerical visual features.
This being a baseline model, only uses four visual features that was identified.

We use spacy's pretrained language model to generate features for the natural langugae text
