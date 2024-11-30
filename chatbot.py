
import os
import time
import pandas as pd

from intent_recognizer import IntentRecognizer
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# HuggingFace API Key
sec_key = os.getenv("HUGGINGFACE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",  # Use an available model
    temperature=0.7,  # Moved from model_kwargs
    model_kwargs={"max_length": 512},  # Kept other parameters in model_kwargs
    huggingfacehub_api_token=sec_key
)

explain_metric_prompt = PromptTemplate(
    input_variables=["metric_name", "metric_meaning"],
    template="""
You are a helpful assistant.

Explain what {metric_name} means.

Answer:
{metric_meaning}
"""
)

# Create Prompt Template for Comparison
compare_metric_prompt = PromptTemplate(
    input_variables=["player1_name", "player2_name", "metric_name", "metric_meaning", "metric_value1", "metric_value2"],
    template="""
You are a cricket analyst assistant.

Compare the {metric_name} of {player1_name} and {player2_name}.

{metric_name}: {metric_meaning}

{player1_name} has a {metric_name} of {metric_value1}.
{player2_name} has a {metric_name} of {metric_value2}.

Explain why one might be better than the other with respect to this metric in less than 50 words.

Answer:
"""
)

compare_metric_chain = compare_metric_prompt | llm
explain_metric_chain = explain_metric_prompt | llm

# Metric meanings dictionary
metric_meanings = {
    'batting_strike_rate': 'Batting strike rate is the number of runs a batsman scores per 100 balls faced.',
    'bowling_economy': 'Bowling economy rate is the average number of runs conceded per over by a bowler.',
    'pitch_score': 'Pitch score represents how favorable the pitch is for the player.',
    'floor': 'Floor value indicates the minimum expected performance of the player.',
    'ceil': 'Ceiling value indicates the maximum potential performance of the player.',
    'batting_first_fantasy_points': 'Predicted fantasy points when the player is batting first.',
    'chasing_first_fantasy_points': 'Predicted fantasy points when the player is chasing.',
    'relative_points': 'Relative points compare the player’s performance to others.',
    'matchup_rank': 'Matchup rank indicates how well the player performs against the current opponent.',
    'six_match_predictions': 'Predictions for the player’s performance in the last six matches.',
    'risk':'Risk tells you how consistent a player is. Low risk means reliable, while high risk could mean unpredictable but high potential!',
    'ai_alert': 'AI-generated alerts regarding the player.'
}

class Chatbot:
    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.player_data = pd.read_csv('mock_data.csv') 
        self.player1_id = None
        self.player1_name = None
        self.player2_id = None
        self.player2_name = None

    def set_players(self, player1_id, player2_id):
        try:
            self.player1_id = int(player1_id)
            self.player2_id = int(player2_id)
        except ValueError:
            print("Invalid player ID. Please enter a numeric ID.")
            return False    
        
        player1_row = self.player_data[self.player_data['player_id'] == self.player1_id]
        player2_row = self.player_data[self.player_data['player_id'] == self.player2_id]
        
        if player1_row.empty:
            print(f"Player ID {self.player1_id} not found in the dataset.")
        else:
            self.player1_name = player1_row['player_name'].values[0]

        if player2_row.empty:
            print(f"Player ID {self.player2_id} not found in the dataset.")
        else:
            self.player2_name = player2_row['player_name'].values[0]

        if player1_row.empty or player2_row.empty:
            available_ids = self.player_data['player_id'].tolist()
            print(f"Available player IDs: {available_ids}")
            return False

        return True

    def get_metric_values(self, metric_name):
        player1_row = self.player_data[self.player_data['player_id'] == self.player1_id]
        player2_row = self.player_data[self.player_data['player_id'] == self.player2_id]

        # Use metric_name (with underscores) to access DataFrame columns
        if metric_name in player1_row.columns and metric_name in player2_row.columns:
            metric_value1 = player1_row[metric_name].values[0]
            metric_value2 = player2_row[metric_name].values[0]
            return metric_value1, metric_value2
        else:
            return None, None



    def process_user_query(self, user_query):
        intent = self.intent_recognizer.recognize_intent(user_query)
        if intent == 'exit':
            return 'exit', None

        if intent is None:
            return None, "I'm sorry, I didn't understand that."

        # Check if the user is asking for an explanation
        explanation_keywords = ['meaning', 'mean', 'explain', 'definition', 'what is']
        if any(keyword in user_query.lower() for keyword in explanation_keywords):
            action = 'explain'
        else:
            action = 'compare'

        # Use intent for data access and metric_name_display for user-friendly output
        metric_name_display = intent.replace('_', ' ')

        # Get metric meaning
        metric_meaning = metric_meanings.get(intent, 'No explanation available.')

        if action == 'compare':
            metric_value1, metric_value2 = self.get_metric_values(intent)
            if metric_value1 is not None and metric_value2 is not None:
                # Prepare inputs for the chain
                chain_inputs = {
                    "player1_name": self.player1_name,
                    "player2_name": self.player2_name,
                    "metric_name": metric_name_display,
                    "metric_meaning": metric_meaning,
                    "metric_value1": metric_value1,
                    "metric_value2": metric_value2
                }
                response = compare_metric_chain.invoke(chain_inputs)
                return intent, response.strip()
            else:
                return intent, f"Data for {metric_name_display} is not available for comparison."
        elif action == 'explain':
            chain_inputs = {
                "metric_name": metric_name_display,
                "metric_meaning": metric_meaning
            }
            response = explain_metric_chain.invoke(chain_inputs)
            return intent, response.strip()
        else:
            return None, "I'm sorry, I didn't understand that."



    def chat(self):
        player1_id = input("Enter Player 1 ID: ")
        player2_id = input("Enter Player 2 ID: ")
        if not self.set_players(player1_id, player2_id):
            return

        print("You can now start chatting with the assistant. Type 'exit' to end the chat.")
        while True:
            user_query = input("You: ")
            if user_query.lower() in ['exit', 'quit']:
                print("Assistant: Goodbye!")
                break

            intent, response = self.process_user_query(user_query)
            if intent == 'exit':
                print("Assistant: Goodbye!")
                break
            elif response:
                print(f"Assistant: {response}")
            else:
                print("Assistant: I'm sorry, I didn't understand that.")

            # Small delay to prevent rate limiting
            time.sleep(0.2)


if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.chat()
