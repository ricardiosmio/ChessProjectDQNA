import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import TensorBoard
import random
from collections import deque
import os
import logging

# Parameters
GAMMA = 0.99
ALPHA = 0.001
MEMORY_SIZE = 100000
BATCH_SIZE = 64
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 1000  # Number of games to play for training

# Configure logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

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
    model.compile(optimizer=Adam(learning_rate=ALPHA), loss='mse')
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

class DQNAgent:
    def __init__(self, state_shape):
        self.state_shape = state_shape
        if os.path.exists('models/trained_model.h5'):
            self.model = load_model('models/trained_model.h5', custom_objects={'mse': mse})
        else:
            self.model = create_model(state_shape)
        self.target_model = create_model(state_shape)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0

        # TensorBoard setup
        self.tensorboard = TensorBoard(log_dir='logs', update_freq='batch')

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
            if best_move is None or q_value > q_values[np.argmax(q_values)]:
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
                target = reward + GAMMA * np.amax(self.target_model.predict(np.expand_dims(encode_board(next_state), axis=0))[0])
            target_f = self.model.predict(np.expand_dims(encode_board(state), axis=0))
            target_f[0][0] = target
            self.model.fit(np.expand_dims(encode_board(state), axis=0), target_f, epochs=1, verbose=0, callbacks=[self.tensorboard])
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, filepath):
        self.model.save(filepath)

    def train_agent(self):
        logging.info("Training started")
        for e in range(EPISODES):
            logging.info(f"Starting episode {e+1}")
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
                    logging.info(f"Episode: {e+1}/{EPISODES}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")
                    break

            self.replay()

            # Save the model periodically
            if e % 100 == 0:
                self.save_model(f'models/model_{e}.h5')

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
    agent.train_agent()