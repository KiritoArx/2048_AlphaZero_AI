import torch
import torch.nn as nn
import torch.nn.functional as F
from game import COLUMN_COUNT, MAX_HEIGHT, PREVIEW_COUNT, Game

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # one channel for board, one for current tile, PREVIEW_COUNT for previews
        self.conv1 = nn.Conv2d(1 + 1 + PREVIEW_COUNT, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.policy_head = nn.Linear(64 * COLUMN_COUNT * MAX_HEIGHT, COLUMN_COUNT)
        self.value_head = nn.Linear(64 * COLUMN_COUNT * MAX_HEIGHT, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value.squeeze(-1)

    def predict(self, game: Game):
        board_tensor = encode_game(game)
        with torch.no_grad():
            policy_logits, value = self.forward(board_tensor)
        policy = policy_logits.squeeze(0).softmax(dim=0).cpu().numpy()
        return policy, float(value.item())

def encode_game(game: Game):
    import numpy as np
    board = np.zeros((1 + 1 + PREVIEW_COUNT, MAX_HEIGHT, COLUMN_COUNT), dtype=np.float32)
    # board channel
    for c in range(COLUMN_COUNT):
        for r, v in enumerate(game.board[c]):
            board[0, r, c] = v
    # current tile channel
    board[1, :, :] = float(game.current)
    # preview channels
    for i, tile in enumerate(list(game.preview)[:PREVIEW_COUNT]):
        board[2 + i, :, :] = float(tile)
    board_tensor = torch.from_numpy(board).unsqueeze(0)
    return board_tensor
