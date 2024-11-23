import json
import random
import torch
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

class ResponseGenerator:
    def __init__(self, intents_file):
        with open(intents_file, 'r') as file:
            self.intents = json.load(file)
        self.lemmatizer = WordNetLemmatizer()

    def get_response(self, intent_tag):
        for intent in self.intents['intents']:
            if intent['tag'] == intent_tag:
                return random.choice(intent['responses'])
        return "I'm not sure how to respond to that."

    def generate_local_resources(self, country):
        resources = {
            "USA": [
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741"
            ],
            "UK": [
                "Samaritans: 116 123",
                "Mind Infoline: 0300 123 3393"
            ],
            # Add more countries as needed
        }
        return resources.get(country, ["International resources: BetterHelp, 7 Cups"])