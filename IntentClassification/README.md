## MultiLayer Perceptron - Baseline

The baseline solution can be found in the mlp folder. The jupyter notebook has also been attached.<br/>
Please run the alfred data generator and the mlp training data generator and upload the data to google drive or to colab before using the notebook.<br/>


## RASA
0. Install Rasa and the dependencies for spacy<br/>

    ```
    $ sudo apt update
    $ sudo apt install python3-dev python3-pip
    $ python3 -m venv ./venv
    $ source ./venv/bin/activate
    $ pip install -U pip
    $ pip install rasa==1.10.12
     OR
    $ pip3 install rasa==1.10.12
    $ pip3 install rasa[full]==1.10.12

    Dependencies for Spacy

    $ pip3 install rasa[spacy]==1.10.12
    $ python3 -m spacy download en_core_web_lg
    $ python3 -m spacy link en_core_web_lg en
    
    Dependecies for the file dialog
    
    $ sudo apt-get install python3-tk
   
   ```

1. Download the ALFRED data<br/>
    ```
    ALFRED data can be downloaded at https://github.com/askforalfred/alfred/tree/master/data/json_2.1.0
    
    We used the full dataset for the project (109 GB)
    https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/full_2.1.0.7z
    ```
    
2. Run the alfred data generator<br/>
   ```
   Remember to activate the virtual environment in the Intent Classification folder
   $ source venv/bin/activate
   
   ./IntentClassifier/IntentClassification/preprocess/alfred_training_data_generator.py
   ```
    
3. Run the rasa training data generator<br/>
   ```
   Remember to activate the virtual environment in the Intent Classification folder
   $ source venv/bin/activate
   
   ./IntentClassifier/IntentClassification/rasa_custom/rasa_training_data_generator.py
   ```
 
4. Training the RASA model
    ```
    Before training the rasa model, ensure that the rasa custom components folder path is added to the PYTHONPATH environment variable
    $export PYTHONPATH=${PATH_TO_PROJECT}/IntentClassification/rasa_custom/:$PYTHONPATH
    
    Remember to activate the virtual environment in the Intent Classification folder
    $ source venv/bin/activate
    
    Train the rasa model by typing in the following command at the project context root folder
    $ rasa train
    ```
    
5. Test the rasa model<br/>
    ```
    Run the file at ./IntentClassifier/IntentClassification/rasa_custom/rasa_model_tester.py
    This program prompts you to input the trained model tar.gz file and the test file
    
    The model file can be found in the ./models folder of the project after the training is completed
    The test file can be found inside the ./data-test folder
    ```

6. To test the rasa model manually using RASA shell (typing in one sample at a time)

    (i)   Bring up the rasa shell window with the appropriate model file
          rasa shell nlu -m models/20201027-211357.tar.gz
        
    (ii)  Enter the test sample - LANGUAGE ONLY
    
          e.g. 
               INPUT                                                                        INTENT
               Pick the book from the corner of the shelf                                   PickupObject            
               hang a right at the wooden dresser and walk to the brown chair ahead         GotoLocation
               go straight and pick up the book                                             GotoLocation PickupObject
               walk straight and put the tea cup down                                       GotoLocation PutObject   
         
    (iii) Enter the test sample - LANGUAGE AND VISUAL INFO

          e.g.
          
            pick up the credit card on the table @@@@@@ 0.21 3.76 6.87 0.87
            Actual Intent --> PickupObject

            pick up the credit card on the table @@@@@@ 0.21 3.76 6.87 0.34
            Actual Intent --> RotateAgent PickupObject"

            pick up the credit card on the table @@@@@@ 0.89 3.76 6.87 0.87
            Actual Intent --> GotoLocation PickupObject
            
            pick up the credit card on the table and place it on the counter @@@@@@ 0.89 3.76 6.87 0.87
            Actual Intent --> GotoLocation PickupObject GotoLocation PutObject
            
            pick up the credit card on the table and place it on the counter @@@@@@ 0.39 3.76 6.87 0.42
            Actual Intent --> RotateAgent PickupObject GotoLocation PutObject
            
            pick up the credit card on the table and place it on the counter @@@@@@ 2.9 3.76 0.34 0.42
            Actual Intent --> GotoLocation PickupObject PutObject
            
            pick up the credit card on the table and place it on the counter @@@@@@ 0.2 0.76 0.34 0.42
            Actual Intent --> RotateAgent PickupObject GotoLocation PutObject
            
            pick up the credit card on the table and place it on the counter @@@@@@ 0.2 0.76 0.34 0.78
            Actual Intent --> PickupObject GotoLocation PutObject
            
            pick up the credit card on the table and place it on the counter @@@@@@ 0.2 0.16 0.34 0.58
            Actual Intent --> PickupObject PutObject
            
            pick up the credit card from the brown couch turn right to reach the smaller brown armchair set the credit card down on the smaller brown armchair @@@@@@ 0.46 1.97 1.96 0.85
            Actual Intent --> PickupObject GotoLocation PutObject
            
            walk quickly to the table @@@@@@ -1 -1 -1 0
            Actual Intent --> GotoLocation
            
            go up to the table in front of you pick up the credit card to the left of the cardboard box on the table turn to your left and walk into the living room then turn to the first arm chair on your right @@@@@@ 0.79 2.0 2.67 0.978
            Actual Intent --> GotoLocation PickupObject GotoLocation
            
            
            put the credit card on the left side of the chair in front of the pillow @@@@@@ 0.23 0.46 7.43 0.86
            Actual Intent --> PutObject
            
            put the credit card on the left side of the chair in front of the pillow @@@@@@ 0.23 0.46 7.43 0.31
            Actual Intent --> RotateAgent PutObject
            
            put the credit card on the left side of the chair in front of the pillow @@@@@@ 0.23 1.46 7.43 0.86
            Actual Intent --> GotoLocation PutObject


7. To test a single instance programmatically,
    ```
    Remember to activate the virtual environment in the Intent Classification folder
    $ source venv/bin/activate
   
    Run the pos_to_rasa method in IntentClassification/rasa_custom/rasa_single_instance_tester.py
    
    The RASA server is assumed to be running with the correct model deployed
    Use the below command to start the server,
    rasa run --enable-api -m models/20201122-221841.tar.gz
    Replace with the appropriate model file
    ````
# preprocess
Contains the code for generating the training data and for creating the training set in the format that RASA understands

# rasa_custom
This folder contains all the custom components written for the customized DIET architecture

# data-train 
Contains the training data input and the validation set

# data-test
Contains the testing data set

# data
Default data folder of rasa. Stores the nlu.json file that RASA uses for training




