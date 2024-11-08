import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tkinter as tk
import chess
from PIL import Image, ImageTk

class PromoteTestGUI:
    LIGHT_SQUARE_COLOR = "#F0D9B5"
    DARK_SQUARE_COLOR = "#B58863"

    def __init__(self, root):
        self.root = root
        self.root.title("Promote Test GUI")

        self.board = chess.Board(None)
        self.board.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
        self.board.set_piece_at(chess.A7, chess.Piece(chess.PAWN, chess.BLACK))
        
        self.board_frame = tk.Frame(root)
        self.board_frame.pack(side=tk.LEFT)

        self.board_canvas = tk.Canvas(self.board_frame, width=480, height=480)
        self.board_canvas.grid(row=1, column=1, rowspan=8, columnspan=8)

        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.RIGHT)

        self.white_button = tk.Button(self.control_frame, text="White's Turn", command=self.play_white)
        self.white_button.pack(side=tk.TOP)

        self.black_button = tk.Button(self.control_frame, text="Black's Turn", command=self.play_black)
        self.black_button.pack(side=tk.TOP)

        self.piece_images = self.load_piece_images()
        self.selected_square = None
        self.is_white_player = True  # By default, the player is white
        self.update_board()

        self.board_canvas.bind("<Button-1>", self.on_square_click)

    def load_piece_images(self):
        pieces = {
            'r': 'DR', 'n': 'DN', 'b': 'DB', 'q': 'DQ', 'k': 'DK', 'p': 'DP',
            'R': 'WR', 'N': 'WN', 'B': 'WB', 'Q': 'WQ', 'K': 'WK', 'P': 'WP'
        }
        piece_images = {}
        for piece, filename in pieces.items():
            image = Image.open(f"images/{filename}.png")
            piece_images[piece] = ImageTk.PhotoImage(image.resize((60, 60)))
        return piece_images

    def play_white(self):
        self.is_white_player = True
        self.update_board()

    def play_black(self):
        self.is_white_player = False
        self.update_board()

    def update_board(self):
        self.board_canvas.delete("all")
        for square in chess.SQUARES:
            x = (square % 8) * 60
            y = (7 - square // 8) * 60 if self.is_white_player else (square // 8) * 60
            color = self.LIGHT_SQUARE_COLOR if (square + square // 8) % 2 == 0 else self.DARK_SQUARE_COLOR
            self.board_canvas.create_rectangle(x, y, x + 60, y + 60, fill=color)

            piece = self.board.piece_at(square)
            if piece is not None:
                piece_image = self.piece_images.get(piece.symbol())
                if piece_image:
                    self.board_canvas.create_image(x, y, image=piece_image, anchor=tk.NW)

    def on_square_click(self, event):
        col = event.x // 60
        row = 7 - (event.y // 60) if self.is_white_player else event.y // 60
        square = chess.square(col, row)

        if self.selected_square is None:
            self.selected_square = square
        else:
            move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                if self.board.piece_at(self.selected_square).piece_type == chess.PAWN and chess.square_rank(square) in [0, 7]:
                    self.promote_pawn(move)
                else:
                    self.board.push(move)
                    self.update_board()
            self.selected_square = None
            self.update_board()

    def promote_pawn(self, move):
        promotion_window = tk.Toplevel(self.root)
        promotion_window.title("Promote Pawn")
        promotion_window.geometry("200x100")

        def set_promotion(piece):
            move.promotion = piece
            self.board.push(move)
            promotion_window.destroy()
            self.update_board()

        tk.Button(promotion_window, text="Queen", command=lambda: set_promotion(chess.QUEEN)).pack()
        tk.Button(promotion_window, text="Rook", command=lambda: set_promotion(chess.ROOK)).pack()
        tk.Button(promotion_window, text="Bishop", command=lambda: set_promotion(chess.BISHOP)).pack()
        tk.Button(promotion_window, text="Knight", command=lambda: set_promotion(chess.KNIGHT)).pack()

if __name__ == "__main__":
    root = tk.Tk()
    gui = PromoteTestGUI(root)
    root.mainloop()
