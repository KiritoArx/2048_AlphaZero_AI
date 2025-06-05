import torch
import torch.nn as nn
import torch.nn.functional as F
from game import COLUMN_COUNT, MAX_HEIGHT, PREVIEW_COUNT, Game


class ResidualBlock(nn.Module):
    """Small residual block with dropout for regularization."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        return F.relu(out + x)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # one channel for board, one for current tile, PREVIEW_COUNT for previews
        self.conv_in = nn.Conv2d(1 + 1 + PREVIEW_COUNT, 64, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.policy_head = nn.Linear(64 * COLUMN_COUNT * MAX_HEIGHT, COLUMN_COUNT)
        self.value_head = nn.Linear(64 * COLUMN_COUNT * MAX_HEIGHT, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv_in(x))
        x = self.res1(x)
        x = self.res2(x)
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

    def predict_batch(self, games: list[Game]):
        """Evaluate a list of games in a single forward pass."""
        batch = torch.cat([encode_game(g) for g in games], dim=0)
        with torch.no_grad():
            policy_logits, values = self.forward(batch)
        policies = policy_logits.softmax(dim=1).cpu().numpy()
        return policies, values.cpu().numpy().tolist()

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
