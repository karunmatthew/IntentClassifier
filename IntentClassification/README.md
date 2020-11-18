README file for Intent Classification


## MultiLayer Perceptron - Baseline

The baseline solution can be found in the mlp folder. The jupyter notebook has also been attached.
Please run the alfred data generator and the mlp training data generator and upload the data to google drive or to colab before using the notebook.


## RASA
0. Install Rasa and the dependencies for spacy

    $ sudo apt update <br/>
    $ sudo apt install python3-dev python3-pip <br/>
    $ python3 -m venv ./venv <br/>
    $ source ./venv/bin/activate <br/>
    $ pip install -U pip <br/>
    $ pip install rasa <br/>
     OR <br/>
    $ pip3 install rasa <br/>

    Dependencies for Spacy

    $ pip3 install rasa[spacy] <br/>
    $ python3 -m spacy download en_core_web_lg <br/>
    $ python3 -m spacy link en_core_web_lg en <br/>



1. Download the ALFRED data
    ALFRED data can be downloaded at https://github.com/askforalfred/alfred/tree/master/data/json_2.1.0
    
    We used the full dataset for the project (109 GB)
    https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/full_2.1.0.7z
    
    
2. Run the alfred data generator
    
3. Run the rasa training data generator

4. Train the rasa model
    $ rasa train
    
5. Test the rasa model
    The rasa model should be loaded and run before the rasa-test is started.


## PREPROCESS

Contains the code for generating the training data and for creating the training set in the format that RASA understands


# data-train 
Contains the training data input and the validation set

# data-test
Contains the testing data set

# data
Default data folder of rasa. Stores the nlu.json file that RASA uses for training




