import random
from collections import deque
from typing import List

COLUMN_COUNT = 5
MAX_HEIGHT = 6
PREVIEW_COUNT = 3

# Board is list of columns, each column is list of ints bottom-up
Board = List[List[int]]

def empty_board() -> Board:
    return [[] for _ in range(COLUMN_COUNT)]

class Game:
    """Dropping-tile merge game state."""
    def __init__(self):
        self.board: Board = empty_board()
        self.preview: deque[int] = deque(self.random_tile() for _ in range(PREVIEW_COUNT))
        self.current: int = self.preview.popleft()
        self.score: int = 0

    def random_tile(self) -> int:
        return random.choice([2, 4])

    def clone(self) -> "Game":
        g = Game.__new__(Game)
        g.board = [col[:] for col in self.board]
        g.preview = deque(self.preview)
        g.current = self.current
        g.score = self.score
        return g

    def reset(self):
        self.board = empty_board()
        self.preview = deque(self.random_tile() for _ in range(PREVIEW_COUNT))
        self.current = self.preview.popleft()
        self.score = 0

    def valid_actions(self) -> List[int]:
        return [c for c in range(COLUMN_COUNT) if len(self.board[c]) < MAX_HEIGHT]

    def step(self, action: int):
        assert 0 <= action < COLUMN_COUNT, "invalid column"
        if len(self.board[action]) >= MAX_HEIGHT:
            raise ValueError("column full")
        prev = self.total_value()
        drop_and_resolve(self.board, self.current, action)
        self.current = self.preview.popleft()
        self.preview.append(self.random_tile())
        self.score = self.total_value()
        reward = self.score - prev
        done = game_over(self.board)
        return reward, done

    def total_value(self) -> int:
        return sum(sum(col) for col in self.board)

    def render(self):
        print_board(self.board)
        print(f"Current: {self.current} Preview: {list(self.preview)} Score: {self.score}\n")


def print_board(board: Board):
    for row in range(MAX_HEIGHT - 1, -1, -1):
        for col in range(COLUMN_COUNT):
            if row < len(board[col]):
                print(f"{board[col][row]:6d}", end="")
            else:
                print("      .", end="")
        print()
    print("-" * COLUMN_COUNT * 6)
    for col in range(COLUMN_COUNT):
        print(f"  C{col:2d} ", end="")
    print("\n")


def drop_and_resolve(board: Board, value: int, col: int):
    from collections import deque

    board[col].append(value)
    row = len(board[col]) - 1

    landing_col = col
    landing_row = row

    fresh = deque([(col, row)])
    while fresh:
        c, r = fresh.popleft()
        if r >= len(board[c]):
            continue
        v = board[c][r]

        # vertical-down
        if r > 0 and board[c][r-1] == v:
            board[c][r-1] *= 2
            del board[c][r]
            fresh.append((c, r-1))
            if r < len(board[c]):
                fresh.append((c, r))
            continue

        # vertical-up
        if r + 1 < len(board[c]) and board[c][r+1] == v:
            board[c][r] *= 2
            del board[c][r+1]
            fresh.append((c, r))
            if r + 1 < len(board[c]):
                fresh.append((c, r+1))
            continue

        # horizontal
        for dc in (-1, 1):
            nc = c + dc
            if nc < 0 or nc >= COLUMN_COUNT:
                continue
            if r < len(board[nc]) and board[nc][r] == v:
                keep_right = (r == landing_row and nc == landing_col)
                dst = nc if keep_right else c
                src = c if keep_right else nc
                board[dst][r] *= 2
                del board[src][r]
                fresh.append((dst, r))
                if r < len(board[src]):
                    fresh.append((src, r))
                break


def game_over(board: Board) -> bool:
    return any(len(col) >= MAX_HEIGHT for col in board)
