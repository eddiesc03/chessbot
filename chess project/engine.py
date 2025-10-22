#all code related to stockfish here
from stockfish import Stockfish
import chess
import time
import matplotlib.pyplot as plt
import random

BASE_ELO = 2500#2000
ELO_VARIANCE = 300  # +/- range
RESIGNATION_THRESHOLD = 500

# Set up Stockfish
stockfish = Stockfish(path="stockfish/stockfish-windows-x86-64-avx2.exe",
    parameters={
        "UCI_LimitStrength": True,
        "UCI_Elo": BASE_ELO  # <-- Set to desired ELO
    }
)

# Global board object that keeps game state
board = chess.Board()
fen_history = [stockfish.get_fen_position()]

def set_varying_elo(stockfish):
    new_elo = BASE_ELO + random.randint(-ELO_VARIANCE, ELO_VARIANCE)
    stockfish.set_elo_rating(new_elo)

def plot_board(board, previous_board):
    plt.close('all')
    unicode_pieces = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    colors = ['#b58863', '#f0d9b5']
    
    # Draw board squares
    for x in range(8):
        for y in range(8):
            color = colors[(x + y) % 2]
            rect = plt.Rectangle((x, y), 1, 1, facecolor=color)
            ax.add_patch(rect)

    # Highlight squares that changed (both white & black pieces)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        prev_piece = previous_board.piece_at(square)

        if piece != prev_piece:
            x = chess.square_file(square)
            y = chess.square_rank(square)

            # Draw a highlighted yellow square
            highlight = plt.Rectangle((x, y), 1, 1, facecolor='yellow', alpha=0.5)
            ax.add_patch(highlight)

    # Draw pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            x = chess.square_file(square)
            y = chess.square_rank(square)
            ax.text(x + 0.5, y + 0.4, unicode_pieces[piece.symbol()],
                    fontsize=48, ha='center', va='center')
            
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show(block = False)
    plt.pause(0.1)

def move(human_move):
    global board  # update global game state

    try:
        move_obj = chess.Move.from_uci(human_move)
    except ValueError:
        print("❌ Invalid move format")
        return

    if move_obj not in board.legal_moves:
        print("❌ Illegal move")
        return

    # Apply human move
    previous_board = chess.Board(board.fen())
    board.push(move_obj)
    stockfish.set_fen_position(board.fen())
    print("✔ Human move:", human_move)
    
    # Check if engine should resign (if it is losing by more than 500 centipawns)
    eval_score = stockfish.get_evaluation()
    if eval_score["type"] == "cp" and eval_score["value"] > RESIGNATION_THRESHOLD:
        print("Stockfish resigns. You win!")
        board.result = "1-0"
       # return
    else:
        # Engine move
        set_varying_elo(stockfish)
        engine_move = stockfish.get_best_move()
        if engine_move:
            board.push(chess.Move.from_uci(engine_move))
            stockfish.make_moves_from_current_position([engine_move])
            fen_history.append(stockfish.get_fen_position())
            print("Stockfish replies:", engine_move)
        else:
            #no move found => game over
            outcome = board.outcome()
            result = outcome.result() if outcome else "?"
            print(f"Game Over: Result = {result}")
            if outcome:
                if result == "1-0":
                    print("You won!")
                elif result == "0-1":
                    print("You lost.")
                else:
                    print("Draw.")
           # save_game_pgn()
            #return



    plot_board(board, previous_board)
   # print(board)
    
def undo_move():
#     fen_history.pop()
#     previous_fen = fen_history[-1]
#     stockfish.set_fen_position(previous_fen)
    global board
    print("undoing")
    board.pop()
    board.pop()
    stockfish.set_fen_position(board.fen())
    plot_board(board)

def get_board():#convert a string to a simple board format
    converted_board = []

    for line in str(board).strip().split('\n'):
        row = []
        for piece in line.split():
            if piece == '.':
                row.append('e')
            elif piece.islower():
                row.append('b')
            elif piece.isupper():
                row.append('w')
        row = row[::-1]
        converted_board.append(row)
    return converted_board
    
#move('e2e4')
# time.sleep(2)
# move('a2a4')