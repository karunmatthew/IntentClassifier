README file for Intent Classification


## BASELINE SOLUTION

The baseline solution can be found in mlp folder. The baseline solution uses a set of generic rules to identify the intent


## RASA

The rasa model should be loaded and run before the rasa-test is started.


## PREPROCESS

Contains the code for generating the training data and for creating the training set in the format that RASA understands


## data-train 

Contains the training data input



$ sudo apt update
$ sudo apt install python3-dev python3-pip
$ python3 -m venv ./venv
$ source ./venv/bin/activate
$ pip install -U pip
$ pip install rasa
