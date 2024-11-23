from flask import Flask, render_template, request, jsonify
from model import MentalHealthChatbotModel, IntentClassifier
from response_generator import ResponseGenerator
import torch
import nltk
import json

# Initialize Flask application
app = Flask(__name__)

# Initialize intent classifier and response generator
intent_classifier = IntentClassifier()
response_generator = ResponseGenerator('intents.json')

# Global dictionary to store conversation contexts for different users
conversation_context = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    
    # Extract user message from request
    user_message = request.json['message']
    
    # Get unique user identifier (default to 'default_user' if not provided)
    user_id = request.json.get('user_id', 'default_user')
    
    # Initialize or retrieve conversation context
    context = conversation_context.get(user_id, {
        'current_intent': None,
        'previous_intent': None,
        'conversation_stage': 'initial'
    })
    
    # Process the conversation and generate response
    response = process_conversation(user_message, context, user_id)
    
    # Update global conversation context
    conversation_context[user_id] = context
    
    # Return response and context
    return jsonify({
        'response': response,
        'context': context
    })

def process_conversation(message, context, user_id):
    
    # Normalize message for processing
    message_lower = message.lower().strip()
    
    # Handle responses awaiting confirmation
    if context['conversation_stage'] == 'awaiting_confirmation':
        # Check for positive confirmation
        if message_lower in ['yes', 'y', 'yeah', 'yep']:
            return handle_confirmation(context)
        
        # Check for negative confirmation
        elif message_lower in ['no', 'n', 'nope']:
            return handle_rejection(context)
    
    # Predict intent for new message
    intent = predict_intent(message)
    
    # Update context intents
    context['previous_intent'] = context['current_intent']
    context['current_intent'] = intent
    
    # Generate context-aware response
    response = generate_context_aware_response(intent, context)
    
    return response

# Load resources at the start of the application
with open('resources.json', 'r') as f:
    MENTAL_HEALTH_RESOURCES = json.load(f)

def handle_confirmation(context):
    
    confirmation_responses = {
        'anxiety': {
            'message': "Great! Let's explore anxiety management techniques. Here are some helpful resources:",
            'resources': MENTAL_HEALTH_RESOURCES['anxiety_resources']
        },
        'depression': {
            'message': "Wonderful. Professional support can help. Check out these resources:",
            'resources': MENTAL_HEALTH_RESOURCES['depression_resources']
        },
        'stress_management': {
            'message': "Excellent! Here are some stress reduction resources:",
            'resources': MENTAL_HEALTH_RESOURCES['stress_management_resources']
        }
    }
    # Prepare response with resources
    response_data = confirmation_responses.get(context['current_intent'], {})
    
    if response_data:
        # Compile response message
        full_response = response_data['message'] + "\n\n"
        
        # Add resource details
        for resource in response_data['resources']:
            full_response += f"- {resource['name']}: {resource['description']}\n"
            if 'website' in resource:
                full_response += f"  Website: {resource['website']}\n"
            if 'phone' in resource:
                full_response += f"  Contact: {resource['phone']}\n"
        
        return full_response
    
    # Fallback response
    return "Confirmed. How else can I support you today?"

def handle_rejection(context):
    
    rejection_responses = {
        'anxiety': "That's okay. Mental health support is a personal journey.",
        'depression': "Your comfort is paramount. Let's explore alternatives.",
        'stress_management': "No problem. We'll find approaches that work for you."
    }
    
    # Return specific or generic rejection response
    return rejection_responses.get(
        context['current_intent'], 
        "I understand. Your well-being is the priority."
    )

def generate_context_aware_response(intent, context):
    
    # Get base response for the intent
    base_response = response_generator.get_response(intent)
    
    # Define follow-up responses for specific intents
    follow_up_responses = {
        'anxiety': "Would you like to learn some coping strategies?",
        'depression': "Shall we discuss support options?",
        'stress_management': "Are you interested in stress reduction techniques?"
    }
    
    # Add follow-up question if applicable
    if intent in follow_up_responses:
        base_response += " " + follow_up_responses[intent]
        context['conversation_stage'] = 'awaiting_confirmation'
    
    return base_response

def predict_intent(message):
    
    # Define keywords for intent classification
    intent_keywords = {
        'loneliness': ['alone', 'lonely', 'isolated', 'homesick'],
        'anxiety': ['anxious', 'worry', 'panic', 'nervous'],
        'depression': ['sad', 'depressed', 'hopeless', 'unhappy'],
        'stress_management': ['stressed', 'overwhelmed', 'burnout'],
        'mental_health_resources': ['help', 'support', 'counseling'],
        'greeting': ['hi', 'hello', 'hey', 'greetings']
    }
    
    # Tokenize message
    message_words = nltk.word_tokenize(message.lower())
    
    # Find matching intent based on keywords
    for intent, keywords in intent_keywords.items():
        if any(keyword in message_words for keyword in keywords):
            return intent
    
    # Default to greeting if no intent detected
    return 'greeting'

# Run the Flask application in debug mode
if __name__ == '__main__':
    app.run(debug=True)