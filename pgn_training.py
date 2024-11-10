import chess.pgn
import os
import logging
import random
from tensorflow.keras.models import load_model
from chess_engine import DQNAgent, encode_board, evaluate_board

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class PGNTrainAgent:
    def __init__(self, model):
        self.model = model
        self.memory = []

    def load_games_from_pgn(self, pgn_file):
        games = []
        with open(pgn_file) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games.append(game)
        logging.info(f"Loaded {len(games)} games from {pgn_file}")
        return games

    def process_game(self, game):
        board = game.board()
        states = []
        for move in game.mainline_moves():
            board.push(move)
            encoded_state = encode_board(board)
            reward = evaluate_board(board)
            states.append((encoded_state, reward))
        return states

    def load_and_process_pgn_files(self, directory):
        all_states = []
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        pgn_files = [f for f in os.listdir(directory) if f.endswith(".pgn")]
        total_games = 0
        for filename in pgn_files:
            games = self.load_games_from_pgn(os.path.join(directory, filename))
            total_games += len(games)
            for game in games:
                states = self.process_game(game)
                all_states.extend(states)
        return all_states, total_games

    def train(self, pgn_directory, models_directory):
        # Load and process PGN files
        states, total_games = self.load_and_process_pgn_files(pgn_directory)
        
        # Load the most trained model
        model_files = [f for f in os.listdir(models_directory) if f.endswith('.keras')]
        if not model_files:
            raise ValueError("No trained model found. Please ensure a trained model is available.")
        
        model_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]), reverse=True)
        latest_model_path = os.path.join(models_directory, model_files[0])
        
        # Log the model loaded for training
        logging.info(f"Loaded model for training: {latest_model_path}")
        
        # Initialize the DQNAgent with the latest model
        agent = DQNAgent(state_shape=(64, 12))
        agent.model = load_model(latest_model_path)
        agent.target_model = agent.model
        
        # Add PGN states to the memory
        for state, reward in states:
            agent.remember(state, random.choice(list(chess.Board().legal_moves)), reward, state, done=False)
        
        # Train the agent using the PGN states
        logging.info(f"Starting training session for {total_games} games from PGN files")
        agent.train_agent(total_games)
        
        # Save the updated model with the new naming convention
        new_model_name = f"PGN{total_games}_{os.path.basename(latest_model_path)}"
        new_model_path = os.path.join(models_directory, new_model_name)
        agent.save_model(new_model_path)
        logging.info(f"Model saved as {new_model_path}")

if __name__ == "__main__":
    pgn_directory = "/Users/maciejkordecki/Documents/GitHub/ChessProjectDQNA/PGN"
    models_directory = "/Users/maciejkordecki/Documents/GitHub/ChessProjectDQNA/models"
    agent = PGNTrainAgent(None)
    agent.train(pgn_directory, models_directory)
