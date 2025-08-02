# from intent import IntentMatcher

# intents = {
#     "logout": [["log", "out"], ["sign", "off"], ["log", "off"]],
#     "shutdown": [["shut", "down"], ["power", "off"], ["turn", "off"]],
#     "power": [["turn", "on"], ["power", "up"], ["start"], ["switch", "on"]],
#     "signup": [["sign", "up"], ["register"], ["create", "account"]],
#     "login": [["log", "in"], ["sign", "in"], ["access", "account"]],
#     "set_temp": [[{"LEMMA": "set"}, {"LEMMA": "the", "OP": "?"}, {"LEMMA": "temperature"}], ["make", "it"], ["set", "it"]],
# }


# # Instantiate once
# matcher = IntentMatcher(intents=intents)

# # Test examples
# examples = [
#     "Turn off the lights",
#     "Shut down my laptop",
#     "Sign off now",
#     "Set temperature to 22 units",
#     "Make it 18",
#     "Just browsing"
# ]

# for text in examples:
#     result = matcher.detect_intent_and_slots(text)
#     print(f"Input: {text}\n → {result}\n")



import os
import pandas as pd
import json

directory = 'D:/Programming/Mimi-VA/test/intent_system/'

for file in os.listdir(directory):
    path = os.path.join(directory, file)
    print(path)
    if file.startswith('skill_'):
        try: 
            with open(path + "/intents.json", "r") as intents:
                print(intents.read())

        except:
            print("could not find intents")
                



### TODO ###
#
# - Get all folders in skills folder starting with skill_
# - Get Json files from said folders
# 