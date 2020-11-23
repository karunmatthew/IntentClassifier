# Author : Karun Mathew
# Student Id: 1007247

# This program posts a single command to the RASA server for intent classification

# PRE-REQUISITES
# This program expects the RASA server to be running at the location specified in apputil.RASA_SERVER
#
# To start a RASA Server, please run the below command
# rasa run --enable-api -m models/20201122-221841.tar.gz
# Please replace the above command with the appropriate trained model

import requests
import json

from util.apputil import RASA_SERVER

headers = {
    'Content-type': 'application/json'
}


# posts a single command instance to the RASA server
# the data includes both language and visual data
# delimited by the LANG_VISUAL_DELIMITER specified in apputil
def post_to_rasa(command):

    data = '{"text": "' + command + '"}'
    response = requests.post(RASA_SERVER, headers=headers, data=data)
    response_json = json.loads(response.text)

    intent = response_json['intent']['name']
    confidence = response_json['intent']['confidence']
    intent_ranking = response_json['intent_ranking']

    print('\nCommand          : ', command)
    print('Predicted Intent : ', intent)
    print('Confidence Score : ', confidence)
    print('Intent Rankings  : ', intent_ranking)


# Sample test instances

# With only language data
post_to_rasa('hang a right at the wooden dresser and walk to the brown chair ahead')

# With both language and visual data
post_to_rasa('pick up the credit card on the table @@@@@@ 0.21 3.76 6.87 0.87')
post_to_rasa('go up to the table in front of you pick up the credit card to the left of the cardboard box on the table turn to your left and walk into the living room then turn to the first arm chair on your right @@@@@@ 0.79 2.0 2.67 0.978')