import math
import random
import tkinter as tk
from tkinter import messagebox
from typing import List, Optional, Tuple

Board = List[str]
HUMAN = None
AI = None


def idx(r: int, c: int, n: int) -> int:
    return r * n + c


def check_winner(board: Board, n: int, k: int) -> Optional[str]:
    # directions: right, down, diag down-right, diag down-left
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(n):
        for c in range(n):
            s = board[idx(r, c, n)]
            if s == ' ':
                continue
            for dr, dc in directions:
                rr, cc = r, c
                ok = True
                for t in range(1, k):
                    rr += dr
                    cc += dc
                    if not (0 <= rr < n and 0 <= cc < n):
                        ok = False
                        break
                    if board[idx(rr, cc, n)] != s:
                        ok = False
                        break
                if ok:
                    return s
    return None


def is_full(board: Board) -> bool:
    return all(cell != ' ' for cell in board)


def available_moves(board: Board) -> List[int]:
    return [i for i, v in enumerate(board) if v == ' ']


def minimax(board: Board, player_to_move: str, n: int, k: int, alpha: int, beta: int, depth: int, max_depth: int) -> Tuple[int, Optional[int]]:
    winner = check_winner(board, n, k)
    if winner is not None:
        if winner == AI:
            return 100000 - (max_depth - depth), None
        return -100000 + (max_depth - depth), None
    if is_full(board):
        return 0, None

    if depth >= max_depth:
        # Heuristic evaluation for 5x5: count open lines of length k-1 and k-2 for both sides.
        return heuristic_score(board, n, k), None

    moves = available_moves(board)

    # order moves: center-ish first to speed pruning
    center = (n - 1) / 2
    def move_priority(m: int) -> float:
        r, c = divmod(m, n)
        return (abs(r - center) + abs(c - center))
    moves.sort(key=move_priority)

    if player_to_move == AI:
        best_score = -math.inf
        best_move = None
        for m in moves:
            board[m] = player_to_move
            score, _ = minimax(board, HUMAN, n, k, alpha, beta, depth + 1, max_depth)
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
        score, _ = minimax(board, AI, n, k, alpha, beta, depth + 1, max_depth)
        board[m] = ' '
        if score < best_score:
            best_score = score
            best_move = m
        beta = min(beta, best_score)
        if beta <= alpha:
            break
    return best_score, best_move


def evaluate_line(cells: List[str]) -> int:
    # cells length == k; assign score based on occurrences.
    a = cells.count(AI)
    h = cells.count(HUMAN)
    if a > 0 and h > 0:
        return 0
    if a == 0 and h == 0:
        return 0
    if h == 0:
        # AI-only
        if a == 1:
            return 2
        if a == 2:
            return 8
        if a == 3:
            return 30
        if a == 4:
            return 120
        return 500
    # HUMAN-only
    if a == 0:
        if h == 1:
            return -2
        if h == 2:
            return -8
        if h == 3:
            return -30
        if h == 4:
            return -120
        return -500
    return 0


def heuristic_score(board: Board, n: int, k: int) -> int:
    # sum over all k-length segments in 4 directions
    total = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for r in range(n):
        for c in range(n):
            for dr, dc in directions:
                rr, cc = r, c
                # check segment fits
                end_r = r + (k - 1) * dr
                end_c = c + (k - 1) * dc
                if not (0 <= end_r < n and 0 <= end_c < n):
                    continue
                cells = []
                for t in range(k):
                    cells.append(board[idx(r + t * dr, c + t * dc, n)])
                if ' ' in cells:
                    # only score partially-filled lines (less optimistic)
                    total += evaluate_line(cells)
                else:
                    total += evaluate_line(cells)
    return total


def best_ai_move(board: Board, ai_symbol: str, human_symbol: str, n: int, k: int) -> int:
    global AI, HUMAN
    AI = ai_symbol
    HUMAN = human_symbol

    # Depth-limited minimax for 5x5 to keep runtime reasonable.
    # For 4-in-a-row, depth 4..5 is usually fine.
    max_depth = 4

    _, move = minimax(board, AI, n, k, alpha=-math.inf, beta=math.inf, depth=0, max_depth=max_depth)
    if move is None:
        return random.choice(available_moves(board))
    return move


class TicTacToeGUI5x5:
    def __init__(self, root: tk.Tk, n: int = 5, k: int = 4):
        self.root = root
        self.root.title("Tic Tac Toe 5x5 (K=4) - AI Minimax")

        self.n = n
        self.k = k

        self.mode = None  # 'ai_starts' | 'human_starts' | 'two_players'
        self.board: Board = [' '] * (n * n)

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
        tk.Button(btns, text="3) 2 người đánh", width=18, command=lambda: self.start_game('two_players')).grid(row=1, column=0, padx=6, pady=4)
        tk.Button(btns, text="Thoát", width=18, command=root.destroy).grid(row=1, column=1, padx=6, pady=4)

        self.grid_frame = tk.Frame(root)
        self.grid_frame.pack(padx=12, pady=(0, 12))

        self.cells: List[tk.Button] = []
        for r in range(n):
            for c in range(n):
                i = idx(r, c, n)
                btn = tk.Button(
                    self.grid_frame,
                    text="",
                    font=("Arial", 14, "bold"),
                    width=4,
                    height=2,
                    command=lambda cell_idx=i: self.on_cell_click(cell_idx),
                )
                btn.grid(row=r, column=c, padx=3, pady=3)
                self.cells.append(btn)

        bottom = tk.Frame(root)
        bottom.pack(padx=12, pady=(0, 12))
        tk.Button(bottom, text="Chơi lại", width=20, command=self.reset_for_same_mode).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(bottom, text="Tắt AI / chuyển mode", width=20, command=self.back_to_mode_select).pack(side=tk.LEFT)

        self.back_to_mode_select()

    def back_to_mode_select(self):
        self.mode = None
        self.board = [' '] * (self.n * self.n)
        self.current_player = 'X'
        self.status_var.set("Chọn chế độ ở dưới")
        for i, btn in enumerate(self.cells):
            # Changed text=str(i + 1) to text=" " to remove numbers
            btn.config(text=" ", state=tk.DISABLED, bg="SystemButtonFace")

    def reset_for_same_mode(self):
        if self.mode is None:
            return
        self.board = [' '] * (self.n * self.n)
        for i, btn in enumerate(self.cells):
            # Changed text=str(i + 1) to text=" " to remove numbers
            btn.config(text=" ", state=tk.NORMAL, bg="SystemButtonFace")

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
                self.status_var.set(f"Lượt của bạn ({self.human_symbol}). Nhấn ô trống.")
        else:
            self.status_var.set(f"Lượt của Player ({self.current_player}).")

    def set_cell(self, idx_cell: int, symbol: str):
        self.board[idx_cell] = symbol
        self.cells[idx_cell].config(text=symbol, state=tk.DISABLED, bg="#d9f2ff")

    def disable_click_during_ai(self):
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

    def on_cell_click(self, idx_cell: int):
        if self.mode is None:
            return
        if self.board[idx_cell] != ' ':
            return

        if self.mode in ('ai_starts', 'human_starts'):
            if self.current_player != self.human_symbol:
                return
            self.set_cell(idx_cell, self.human_symbol)
            self.current_player = self.ai_symbol
            self.after_move_check()
            return

        # two players
        self.set_cell(idx_cell, self.current_player)
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        self.after_move_check()

    def after_move_check(self):
        winner = check_winner(self.board, self.n, self.k)
        if winner is not None:
            messagebox.showinfo("Kết thúc", f"{winner} thắng!")
            self.disable_all()
            return
        if is_full(self.board):
            messagebox.showinfo("Kết thúc", "Hòa!")
            self.disable_all()
            return

        self.update_status()
        self.disable_click_during_ai()
        self.maybe_ai_move()

    def disable_all(self):
        for btn in self.cells:
            btn.config(state=tk.DISABLED)

    def maybe_ai_move(self):
        if self.mode not in ('ai_starts', 'human_starts'):
            return
        if self.current_player != self.ai_symbol:
            return
        self.root.after(150, self.do_ai_move)

    def do_ai_move(self):
        if self.mode is None or self.current_player != self.ai_symbol:
            return

        idx_cell = best_ai_move(self.board[:], self.ai_symbol, self.human_symbol, self.n, self.k)
        if self.board[idx_cell] != ' ':
            empty = [i for i, v in enumerate(self.board) if v == ' ']
            if not empty:
                return
            idx_cell = random.choice(empty)

        self.set_cell(idx_cell, self.ai_symbol)
        self.current_player = self.human_symbol
        self.after_move_check()


if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI5x5(root, n=7, k=4)
    root.mainloop()


