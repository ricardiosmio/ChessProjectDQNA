import sys
import math
import chess
import numpy as np
import keras
import tensorflow as tf

def inverse_sigmoid(x):
    return -1 * math.log((1 / x) - 1)

class SimpleChessEngine():
    def __init__(self, board: chess.Board, depth=0):
        self.board = board
        self.depth = depth
        self.recursive_calls = 0
        self.evaluation_type = 'Material'

    def evaluate(self) -> int:
        outcome = self.board.outcome()
        if outcome != None:
            if outcome.winner == None: return 0
            if outcome.winner == self.board.turn: return 9000
            if outcome.winner != self.board.turn: return -9000

        evaluation = 0
        piece_value = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }

        for square in self.board.piece_map():
            piece = self.board.piece_at(square)
            if piece == None: continue
            points = piece_value[piece.piece_type]
            if piece.color == chess.BLACK:
                points *= -1
            evaluation += points

        perspective = 1 if self.board.turn == chess.WHITE else -1
        return evaluation * perspective

    def ordered_moves(self) -> list:
        ordered_list = []
        for move in self.board.legal_moves:
            if move.drop != None or move.promotion != None:
                ordered_list.insert(0, move)
            else:
                ordered_list.append(move)
        return ordered_list

    def nega_max_alpha_beta(self, depth, alpha=-1 * sys.maxsize, beta=sys.maxsize):
        self.recursive_calls += 1
        if depth == 0 or self.ordered_moves() == []: return self.evaluate()
        for move in self.ordered_moves():
            self.board.push(move)
            score = -1 * self.nega_max_alpha_beta(depth=depth-1, alpha=-1 * beta, beta=-1 * alpha)
            self.board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def get_move(self) -> chess.Move:
        self.recursive_calls = 0
        best_move = None
        max_eval = -1 * sys.maxsize
        for move in self.ordered_moves():
            self.board.push(move)
            value = -1 * self.nega_max_alpha_beta(depth=self.depth)
            self.board.pop()
            if value >= max_eval:
                best_move = move
                max_eval = value
        return best_move, max_eval, self.recursive_calls

class AiEngine(SimpleChessEngine):
    def __init__(self, model: str, board: chess.Board, depth=0):
        super().__init__(board, depth)
        self.keras_model = keras.models.load_model(model)
        self.evaluation_type = 'Neural Network Prediction (Keras)'

    def fen_to_image(self, fen_string: str) -> np.array:
        piece_to_channel = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        image = np.zeros([8, 8, 12], dtype='float32')
        parts = fen_string.split(" ")
        fen_board = parts[0].split('/')
        for i, row in enumerate(fen_board):
            j = 0
            for char in row:
                if char in piece_to_channel:
                    val = 1 if char.isupper() else -1
                    image[i, j, piece_to_channel[char]] = val
                    j += 1
                else:
                    j += int(char)
        out = np.empty([1, 8, 8, 12])
        out[0,] = image
        return out

    def evaluate(self) -> float:
        outcome = self.board.outcome()
        if outcome != None:
            if outcome.winner == None: return 0
            if outcome.winner == self.board.turn: return 9000
            if outcome.winner != self.board.turn: return -9000
        fen = self.board.fen()
        image = self.fen_to_image(fen)
        image = image.reshape(1, 64, 12)  # Reshape the image to match the model's expected input
        prediction = self.keras_model.predict(image, verbose=0)[0][0]
        evaluation = 100 * inverse_sigmoid(prediction)
        perspective = 1 if self.board.turn == chess.WHITE else -1
        return evaluation * perspective

class TFLiteEngine(AiEngine):
    def __init__(self, model: str, board: chess.Board, depth=0):
        super().__init__(model, board, depth)
        self.evaluation_type = 'Neural Network Prediction (TfLite)'
        converter = tf.lite.TFLiteConverter.from_saved_model(model)
        tflite_model = converter.convert()
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def evaluate(self) -> float:
        outcome = self.board.outcome()
        if outcome != None:
            if outcome.winner == None: return 0
            if outcome.winner == self.board.turn: return 9000
            if outcome.winner != self.board.turn: return -9000
        fen = self.board.fen()
        input_data = np.float32(self.fen_to_image(fen))
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_index)
        evaluation = 100 * inverse_sigmoid(prediction)
        perspective = 1 if self.board.turn == chess.WHITE else -1
        return evaluation * perspective
