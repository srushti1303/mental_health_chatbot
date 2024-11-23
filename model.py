import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('wordnet')

class MentalHealthChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MentalHealthChatbotModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class IntentClassifier:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_chars = ['?', '!', '.', ',']

    def preprocess_data(self, intents_file):
        with open(intents_file, 'r') as file:
            intents = json.load(file)

        for intent in intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern.lower())
                self.words.extend(word_list)
                self.documents.append((word_list, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(word.lower()) for word in self.words if word not in self.ignore_chars]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

    def create_training_data(self):
        training = []
        output_empty = [0] * len(self.classes)

        for doc in self.documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]

            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)
        
        X = np.array([np.array(row[0]) for row in training])
        y = np.array([np.array(row[1]) for row in training])

        return X, y

    def save_preprocessed_data(self, X, y):
        with open('mental_health_data.pkl', 'wb') as file:
            pickle.dump((X, y, self.words, self.classes), file)