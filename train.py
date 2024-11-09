import chess
import numpy as np
from chess_engine import DQNAgent, encode_board
import logging
import os

# Parameters
EPISODES = 1000  # Number of games to play for training

# Configure logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def train_agent():
    agent = DQNAgent(state_shape=(64, 12))
    for e in range(EPISODES):
        board = chess.Board()
        state = encode_board(board)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(board)
            board.push(action)
            reward = evaluate_board(board)
            next_state = encode_board(board)
            done = board.is_game_over()
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                agent.update_target_model()
                logging.info(f"Episode: {e+1}/{EPISODES}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
                break
        
        agent.replay()
        
        # Save the model periodically
        if e % 100 == 0:
            agent.model.save(f'model_{e}.h5')

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
    train_agent()
