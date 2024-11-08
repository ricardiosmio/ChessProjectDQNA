import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tkinter as tk
import chess
from PIL import Image, ImageTk
from chess_engine import DQNAgent, encode_board  # Import the engine

class ChessGUI:
    LIGHT_SQUARE_COLOR = "#F0D9B5"
    DARK_SQUARE_COLOR = "#B58863"

    def __init__(self, root):
        self.root = root
        self.root.title("Chess GUI")

        self.board = chess.Board()
        self.board_frame = tk.Frame(root)
        self.board_frame.pack(side=tk.LEFT)

        self.board_canvas = tk.Canvas(self.board_frame, width=480, height=480)
        self.board_canvas.grid(row=1, column=1, rowspan=8, columnspan=8)

        self.add_labels()

        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.RIGHT)

        self.white_button = tk.Button(self.control_frame, text="Play as White", command=self.play_white)
        self.white_button.pack(side=tk.TOP)

        self.black_button = tk.Button(self.control_frame, text="Play as Black", command=self.play_black)
        self.black_button.pack(side=tk.TOP)

        self.move_history = tk.Text(self.control_frame, width=30, height=20, state=tk.DISABLED)
        self.move_history.pack(side=tk.TOP)

        self.status_label = tk.Label(self.control_frame, text="", font=("Arial", 14))
        self.status_label.pack(side=tk.TOP)

        self.piece_images = self.load_piece_images()
        self.selected_square = None
        self.is_white_player = True  # By default, the player is white
        self.agent = DQNAgent((64, 12))  # Initialize the agent
        self.update_board()

        self.board_canvas.bind("<Button-1>", self.on_square_click)

    def add_labels(self):
        self.row_labels = []
        self.col_labels = []
        for i in range(8):
            col_label_top = tk.Label(self.board_frame, text=chr(65 + i), font=("Arial", 12))
            col_label_top.grid(row=0, column=i+1)
            self.col_labels.append(col_label_top)

            col_label_bottom = tk.Label(self.board_frame, text=chr(65 + i), font=("Arial", 12))
            col_label_bottom.grid(row=9, column=i+1)
            self.col_labels.append(col_label_bottom)

            row_label_left = tk.Label(self.board_frame, text=str(8 - i), font=("Arial", 12))
            row_label_left.grid(row=i+1, column=0)
            self.row_labels.append(row_label_left)

            row_label_right = tk.Label(self.board_frame, text=str(8 - i), font=("Arial", 12))
            row_label_right.grid(row=i+1, column=9)
            self.row_labels.append(row_label_right)

    def update_labels(self):
        for i, label in enumerate(self.col_labels):
            label.config(text=chr(65 + (i // 2) if self.is_white_player else 65 + (7 - i // 2)))
        for i, label in enumerate(self.row_labels):
            label.config(text=str(8 - (i // 2) if self.is_white_player else 1 + (i // 2)))

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
        self.board = chess.Board()
        self.is_white_player = True
        self.update_board()
        self.update_labels()
        self.update_move_history()
        self.status_label.config(text="")

    def play_black(self):
        self.board = chess.Board()
        self.is_white_player = False
        self.update_board()
        self.update_labels()
        self.update_move_history()
        self.status_label.config(text="")
        self.engine_move()  # Engine makes the first move

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
            self.highlight_moves(square)
            print(f"Selected square: {self.selected_square}")
            print(f"Legal moves: {list(self.board.legal_moves)}")
        else:
            if self.board.piece_at(self.selected_square).piece_type == chess.PAWN and chess.square_rank(square) in [0, 7]:
                self.promote_pawn(self.selected_square, square)
            else:
                move = chess.Move(self.selected_square, square)
                print(f"Attempting move: {move}")

                if move in self.board.legal_moves:
                    print("Move is legal")
                    self.board.push(move)
                    self.update_board()
                    self.update_move_history()
                    self.check_game_status()
                    self.engine_move()
                else:
                    print("Illegal move, resetting selection")

            self.selected_square = None

    def promote_pawn(self, from_square, to_square):
        x, y = self.root.winfo_pointerxy()

        promotion_window = tk.Toplevel(self.root)
        promotion_window.title("Promote Pawn")
        promotion_window.geometry(f"200x100+{x}+{y}")

        pawn_color = 'W' if self.board.piece_at(from_square).color == chess.WHITE else 'D'
        label = tk.Label(promotion_window, text="Promote pawn to:")
        label.pack()

        def set_promotion(piece_type):
            move = chess.Move(from_square, to_square, promotion=piece_type)

            if move in self.board.legal_moves:
                self.board.push(move)
                promotion_window.destroy()
                self.update_board()
                self.update_move_history()
                self.check_game_status()
                self.engine_move()
            else:
                print("Invalid promotion move")
                promotion_window.destroy()

        tk.Button(promotion_window, text="Queen", command=lambda: set_promotion(chess.QUEEN)).pack(side=tk.LEFT)
        tk.Button(promotion_window, text="Rook", command=lambda: set_promotion(chess.ROOK)).pack(side=tk.LEFT)
        tk.Button(promotion_window, text="Bishop", command=lambda: set_promotion(chess.BISHOP)).pack(side=tk.LEFT)
        tk.Button(promotion_window, text="Knight", command=lambda: set_promotion(chess.KNIGHT)).pack(side=tk.LEFT)

    def highlight_moves(self, square):
        legal_moves = list(self.board.legal_moves)
        for move in legal_moves:
            if move.from_square == square:
                self.board_canvas.create_rectangle(
                    (move.to_square % 8) * 60, (7 - move.to_square // 8) * 60 if self.is_white_player else (move.to_square // 8) * 60,
                    (move.to_square % 8 + 1) * 60, (8 - move.to_square // 8) * 60 if self.is_white_player else (move.to_square // 8 + 1) * 60,
                    outline="yellow", width=3
                )

    def update_move_history(self):
        self.move_history.config(state=tk.NORMAL)
        self.move_history.delete('1.0', tk.END)
        move_list = list(self.board.move_stack)
        for i, move in enumerate(move_list):
            if i % 2 == 0:
                self.move_history.insert(tk.END, f"{i//2 + 1}. {move} ")
            else:
                self.move_history.insert(tk.END, f"{move}\n")
        self.move_history.config(state=tk.DISABLED)

    def check_game_status(self):
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn else "White"
            self.status_label.config(text=f"Checkmate! {winner} wins.")
        elif self.board.is_stalemate():
            self.status_label.config(text="Stalemate!")
        elif self.board.is_insufficient_material():
            self.status_label.config(text="Draw by insufficient material.")
        elif self.board.is_seventyfive_moves():
            self.status_label.config(text="Draw by 75-move rule.")
        elif self.board.is_fivefold_repetition():
            self.status_label.config(text="Draw by fivefold repetition.")
        elif self.board.is_variant_draw():
            self.status_label.config(text="Draw!")

    def engine_move(self):
        encoded_board = encode_board(self.board)
        best_move = self.agent.act(self.board)  # Use act instead of choose_action
        if best_move:
            print(f"Engine chose: {best_move}")
            self.board.push(best_move)
            self.update_board()
            self.update_move_history()
            self.check_game_status()
        else:
            print("Engine found no valid move")

def main():
    root = tk.Tk()
    gui = ChessGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
