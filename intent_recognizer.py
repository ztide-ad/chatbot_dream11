import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class IntentRecognizer:
    def __init__(self):
        # Define the intents and their descriptions
        self.intents = {
            'batting_strike_rate': 'Retrieve the batting strike rate (runs per 100 balls) of a player.',
            'bowling_economy': 'Get the bowling economy rate (runs conceded per over) of a player.',
            'pitch_score': 'Provide the pitch score for the player at a venue.',            
            'floor': 'Show the floor value of the player.',
            'ceil': 'Show the ceiling value of the player.',
            'batting_first_fantasy_points': 'Get predicted fantasy points when batting first.',
            'chasing_fantasy_points': 'Obtain predicted fantasy points when the player is batting second that is in chasing.',
            'relative_points': 'Show the relative fantasy points of the player compared to others.',
            'matchup_rank': 'Provide the matchup rank of the player.',
            'six_match_predictions': 'Get for the player’s performance in the last six matches.',
            'risk':'Get the risk factor of the player.',            
            'ai_alert': 'Show AI alerts for the player.'
        }

        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Precompute embeddings
        self.intent_embeddings = {}
        for intent, description in self.intents.items():
            intent_embedding = self.get_embedding(description)
            if intent_embedding.ndim == 1:
                intent_embedding = intent_embedding.reshape(1, -1)
            self.intent_embeddings[intent] = intent_embedding
            
            
    def get_embedding(self, text):
        embedding = self.model.encode(text)
        return embedding.reshape(1, -1)  # Reshape to 2D array



    def recognize_intent(self, user_query):
        query_embedding = self.get_embedding(user_query)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        similarities = {}
        for intent, intent_embedding in self.intent_embeddings.items():
            if intent_embedding.ndim == 1:
                intent_embedding = intent_embedding.reshape(1, -1)
            similarity = cosine_similarity(query_embedding, intent_embedding)[0][0]
            similarities[intent] = similarity
        recognized_intent = max(similarities, key=similarities.get)
        max_similarity = similarities[recognized_intent]
        threshold = 0.6  # Adjust as needed
        if max_similarity >= threshold:
            return recognized_intent
        else:
            return None

