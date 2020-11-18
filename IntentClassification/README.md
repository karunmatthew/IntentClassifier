README file for Intent Classification


## BASELINE SOLUTION

The baseline solution can be found in mlp folder. The baseline solution uses a set of generic rules to identify the intent


## RASA

The rasa model should be loaded and run before the rasa-test is started.


## PREPROCESS

Contains the code for generating the training data and for creating the training set in the format that RASA understands


## data-train 

Contains the training data input



$ sudo apt update <br/>
$ sudo apt install python3-dev python3-pip <br/>
$ python3 -m venv ./venv <br/>
$ source ./venv/bin/activate <br/>
$ pip install -U pip <br/>
$ pip install rasa <br/>
$ pip3 install rasa <br/>

Dependencies for Spacy

$ pip3 install rasa[spacy] <br/>
$ python3 -m spacy download en_core_web_lg <br/>
$ python3 -m spacy link en_core_web_md en <br/>
