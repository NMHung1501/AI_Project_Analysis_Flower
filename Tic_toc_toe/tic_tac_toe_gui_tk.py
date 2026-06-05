import math
import random
import tkinter as tk
from tkinter import messagebox
from typing import List, Optional, Tuple

Board = List[str]  # length 9: 'X','O',' '

HUMAN = None
AI = None


def check_winner(board: Board) -> Optional[str]:
    wins = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]
    for a, b, c in wins:
        if board[a] != ' ' and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_full(board: Board) -> bool:
    return all(cell != ' ' for cell in board)


def available_moves(board: Board) -> List[int]:
    return [i for i, v in enumerate(board) if v == ' ']


def minimax(board: Board, player_to_move: str, alpha: int, beta: int) -> Tuple[int, Optional[int]]:
    winner = check_winner(board)
    if winner is not None:
        if winner == AI:
            return 10, None
        return -10, None
    if is_full(board):
        return 0, None

    moves = available_moves(board)

    # Order moves for faster alpha-beta pruning
    priority = {4: 0, 0: 1, 2: 1, 6: 1, 8: 1, 1: 2, 3: 2, 5: 2, 7: 2}
    moves.sort(key=lambda x: priority.get(x, 99))

    if player_to_move == AI:
        best_score = -math.inf
        best_move = None
        for m in moves:
            board[m] = player_to_move
            score, _ = minimax(board, HUMAN, alpha, beta)
            board[m] = ' '
            if score > best_score:
                best_score = score
                best_move = m
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score, best_move

    best_score = math.inf
    best_move = None
    for m in moves:
        board[m] = player_to_move
        score, _ = minimax(board, AI, alpha, beta)
        board[m] = ' '
        if score < best_score:
            best_score = score
            best_move = m
        beta = min(beta, best_score)
        if beta <= alpha:
            break
    return best_score, best_move


def best_ai_move(board: Board, ai_symbol: str, human_symbol: str) -> int:
    global AI, HUMAN
    AI = ai_symbol
    HUMAN = human_symbol

    _, move = minimax(board, AI, alpha=-math.inf, beta=math.inf)
    if move is None:
        return random.choice(available_moves(board))
    return move


class TicTacToeGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Tic Tac Toe 3x3 - AI Minimax (Tkinter GUI)")

        self.mode = None  # 'ai_starts' | 'human_starts' | 'two_players'
        self.board: Board = [' '] * 9

        self.ai_symbol = 'X'
        self.human_symbol = 'O'
        self.current_player = 'X'

        top = tk.Frame(root)
        top.pack(padx=12, pady=10)

        self.status_var = tk.StringVar(value="Chọn chế độ ở dưới")
        status = tk.Label(top, textvariable=self.status_var, font=("Arial", 12, "bold"))
        status.pack(pady=(0, 8))

        btns = tk.Frame(top)
        btns.pack()

        tk.Button(btns, text="1) AI đánh trước", width=18, command=lambda: self.start_game('ai_starts')).grid(row=0, column=0, padx=6, pady=4)
        tk.Button(btns, text="2) Người đánh trước", width=18, command=lambda: self.start_game('human_starts')).grid(row=0, column=1, padx=6, pady=4)
        tk.Button(btns, text="3) 2 người đánh nhau", width=18, command=lambda: self.start_game('two_players')).grid(row=1, column=0, padx=6, pady=4)
        tk.Button(btns, text="Thoát", width=18, command=root.destroy).grid(row=1, column=1, padx=6, pady=4)

        self.grid_frame = tk.Frame(root)
        self.grid_frame.pack(padx=12, pady=(0, 12))

        self.cells: List[tk.Button] = []
        for r in range(3):
            for c in range(3):
                i = r * 3 + c
                btn = tk.Button(
                    self.grid_frame,
                    text=str(i + 1),
                    font=("Arial", 20, "bold"),
                    width=4,
                    height=2,
                    command=lambda idx=i: self.on_cell_click(idx),
                )
                btn.grid(row=r, column=c, padx=4, pady=4)
                self.cells.append(btn)

        bottom = tk.Frame(root)
        bottom.pack(padx=12, pady=(0, 12))
        tk.Button(bottom, text="Chơi lại", width=20, command=self.reset_for_same_mode).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(bottom, text="Tắt AI / chuyển mode", width=20, command=self.back_to_mode_select).pack(side=tk.LEFT)

        self.back_to_mode_select()

    def back_to_mode_select(self):
        self.mode = None
        self.board = [' '] * 9
        self.current_player = 'X'
        self.status_var.set("Chọn chế độ ở dưới")
        for i, btn in enumerate(self.cells):
            btn.config(text=str(i + 1), state=tk.DISABLED, bg="SystemButtonFace")

    def reset_for_same_mode(self):
        if self.mode is None:
            return
        # keep mode, reset board and replay rules
        self.board = [' '] * 9
        for i, btn in enumerate(self.cells):
            btn.config(text=str(i + 1), state=tk.NORMAL, bg="SystemButtonFace")

        if self.mode == 'ai_starts':
            self.ai_symbol, self.human_symbol = 'X', 'O'
            self.current_player = 'X'
        elif self.mode == 'human_starts':
            self.ai_symbol, self.human_symbol = 'O', 'X'
            self.current_player = 'X'
        else:
            self.ai_symbol = None
            self.human_symbol = None
            self.current_player = 'X'

        self.update_status()
        self.maybe_ai_move()

    def start_game(self, mode: str):
        self.mode = mode
        self.reset_for_same_mode()

    def update_status(self):
        if self.mode in ('ai_starts', 'human_starts'):
            if self.current_player == self.ai_symbol:
                self.status_var.set(f"Đang chờ AI ({self.ai_symbol}) đi...")
            else:
                self.status_var.set(f"Lượt của bạn ({self.human_symbol}). Nhấn ô bất kỳ trống.")
        else:
            self.status_var.set(f"Lượt của Player ({self.current_player}).")

    def set_cell(self, idx: int, symbol: str):
        self.board[idx] = symbol
        self.cells[idx].config(text=symbol, state=tk.DISABLED, bg="#d9f2ff")

    def on_cell_click(self, idx: int):
        if self.mode is None:
            return
        if self.board[idx] != ' ':
            return

        # turn handling
        if self.mode in ('ai_starts', 'human_starts'):
            if self.current_player != self.human_symbol:
                return
            self.set_cell(idx, self.human_symbol)
            self.current_player = self.ai_symbol
            self.after_move_check()
            return

        # two players
        self.set_cell(idx, self.current_player)
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        self.after_move_check()

    def after_move_check(self):
        winner = check_winner(self.board)
        if winner is not None:
            messagebox.showinfo("Kết thúc", f"{winner} thắng!")
            self.disable_all()
            return
        if is_full(self.board):
            messagebox.showinfo("Kết thúc", "Hòa!")
            self.disable_all()
            return

        self.update_status()
        self.maybe_ai_move()

    def disable_all(self):
        for btn in self.cells:
            btn.config(state=tk.DISABLED)

    def maybe_ai_move(self):
        if self.mode not in ('ai_starts', 'human_starts'):
            return
        if self.current_player != self.ai_symbol:
            return

        # make AI move after a short delay so UI updates
        self.root.after(200, self.do_ai_move)

    def do_ai_move(self):
        if self.mode is None or self.current_player != self.ai_symbol:
            return

        idx = best_ai_move(self.board[:], self.ai_symbol, self.human_symbol)
        if self.board[idx] != ' ':
            # should not happen, but safe fallback
            empty = [i for i, v in enumerate(self.board) if v == ' ']
            if not empty:
                return
            idx = random.choice(empty)

        self.set_cell(idx, self.ai_symbol)
        self.current_player = self.human_symbol
        self.after_move_check()

    def disable_non_turn_cells(self):
        # Optional: could be used to prevent click during AI turn
        if self.mode not in ('ai_starts', 'human_starts'):
            return
        if self.current_player == self.ai_symbol:
            for i, btn in enumerate(self.cells):
                if self.board[i] == ' ':
                    btn.config(state=tk.DISABLED)
        else:
            for i, btn in enumerate(self.cells):
                if self.board[i] == ' ':
                    btn.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()

