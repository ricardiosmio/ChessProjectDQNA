import chess
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
import random
from collections import deque
import os
import logging

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Parameters
GAMMA = 0.99
ALPHA = 0.001
MEMORY_SIZE = 100000
BATCH_SIZE = 64
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 10  # Number of games to play for training

# Register mse as a serializable function
tf.keras.utils.register_keras_serializable()(mse)

@tf.keras.utils.register_keras_serializable()
def create_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=ALPHA), loss='mse', metrics=['accuracy'])
    return model

def encode_board(board):
    encoded = np.zeros((64, 12), dtype=int)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.symbol()
            index = chess.square_rank(square) * 8 + chess.square_file(square)
            encoded[index][piece_map[piece_type]] = 1
    return encoded

def get_most_trained_model(directory='models'):
    model_files = [f for f in os.listdir(directory) if f.endswith('.keras')]
    if not model_files:
        return None
    model_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]), reverse=True)
    return os.path.join(directory, model_files[0])

class DQNAgent:
    def __init__(self, state_shape, start_games=100):
        self.state_shape = state_shape
        self.start_games = start_games
        model_path = get_most_trained_model()
        if model_path:
            self.model = load_model(model_path)
            self.model.compile(optimizer=Adam(learning_rate=ALPHA), loss='mse', metrics=['accuracy'])
            self.model.evaluate(np.zeros((1, 64, 12)), np.zeros((1, 1)))  # Ensure metrics are compiled by evaluating the model
        else:
            raise ValueError("No trained model found. Please ensure a trained model is available.")
        self.target_model = self.model
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        # Load epsilon value from file if it exists
        self.epsilon = 1.0
        epsilon_file = 'epsilon_value.txt'
        if os.path.isfile(epsilon_file):
            with open(epsilon_file, 'r') as f:
                self.epsilon = float(f.read())
                
        self.epsilon_min = 0.01  # Minimum epsilon value
        self.epsilon_decay = 0.995  # Decay rate for epsilon

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(list(state.legal_moves))
        q_values = []
        best_move = None
        for move in state.legal_moves:
            state.push(move)
            q_value = self.model.predict(np.expand_dims(encode_board(state), axis=0))[0][0]
            q_values.append(q_value)
            if best_move is None or q_value > max(q_values):
                best_move = move
            state.pop()
        return best_move

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + GAMMA * np.amax(self.target_model.predict(np.expand_dims(next_state, axis=0))[0])
            target_f = self.model.predict(np.expand_dims(state, axis=0))
            target_f[0][0] = target
            self.model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, filepath):
        save_model(self.model, filepath)

    def train_agent(self, num_games):
        logging.info(f"Starting training session for {num_games} games")
        for i in range(num_games):
            logging.info(f"Game {i + 1} started")
            board = chess.Board()
            state = encode_board(board)
            total_reward = 0
            done = False

            while not done:
                action = self.act(board)
                board.push(action)
                reward = evaluate_board(board)
                next_state = encode_board(board)
                done = board.is_game_over()

                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    self.update_target_model()
                    logging.info(f"Game {i + 1} ended, Total Reward: {total_reward}, Epsilon: {self.epsilon}")
                    break

            self.replay()

        # Save the updated epsilon value
        with open('epsilon_value.txt', 'w') as f:
            f.write(str(self.epsilon))

        self.save_model(f"models/trained_model_{self.start_games + num_games}.keras")

def evaluate_board(board):
    if board.is_checkmate():
        return 100 if board.turn == chess.BLACK else -100
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 10000
    }
    evaluation = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values.get(piece.piece_type, 0)
            evaluation += value if piece.color == chess.WHITE else -value
    return evaluation

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    agent = DQNAgent(state_shape=(64, 12))
    agent.train_agent(EPISODES)
