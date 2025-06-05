from collections import deque
import csv
import os
import random
import math
from typing import Optional

import torch
import torch.nn.functional as F

from game import Game, game_over, COLUMN_COUNT
from mcts import MCTS, Node
from net import SimpleNet, encode_game
from evaluate import evaluate


class ReplayBuffer:
    """Simple replay buffer for self-play examples."""

    def __init__(self, capacity: int = 1000):
        self.buffer: deque = deque(maxlen=capacity)

    def add_examples(self, examples):
        self.buffer.extend(examples)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        return list(states), list(policies), list(values)

    def save(self, path: str):
        torch.save(list(self.buffer), path)

    def load(self, path: str):
        if os.path.exists(path):
            data = torch.load(path)
            self.buffer = deque(data, maxlen=self.buffer.maxlen)


def self_play(model, episodes: int = 1, max_moves: int = 50, *, simulations: int = 50,
              temperature_moves: int = 10, log_csv: Optional[csv.writer] = None):
    """Run self-play games and return training examples and game scores."""
    examples = []
    scores = []
    for _ in range(episodes):
        game = Game()
        root = Node(game)
        mcts = MCTS(model, simulations=simulations, batch_size=8)
        trajectory = []
        move = 0
        done = False
        while not done and move < max_moves:
            mcts.run(root)
            visits = {a: child.visit_count for a, child in root.children.items()}
            total = sum(visits.values()) + 1e-8
            policy = [visits.get(a, 0) / total for a in range(COLUMN_COUNT)]
            trajectory.append((game.clone(), policy))

            temp = 1.0 if move < temperature_moves else 0.0
            action = mcts.select_action(root, temperature=temp)
            _, done = game.step(action)
            root = Node(game)
            move += 1

        value = math.tanh(game.score / 1000.0)
        for state, pol in trajectory:
            examples.append((state, pol, value))
        scores.append(game.score)
        if log_csv is not None:
            log_csv.writerow([game.score])
    return examples, scores


def train_step(model: SimpleNet, optimizer, examples, batch_size: int = 32):
    states, policies, values = examples
    inputs = torch.cat([encode_game(s) for s in states], dim=0)
    target_policy = torch.tensor(policies, dtype=torch.float32)
    target_value = torch.tensor(values, dtype=torch.float32)

    policy_logits, value = model(inputs)
    log_policy = F.log_softmax(policy_logits, dim=1)
    loss_policy = -(target_policy * log_policy).sum(dim=1).mean()
    loss_value = F.mse_loss(value.squeeze(-1), target_value)
    loss = loss_policy + loss_value

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def save_checkpoint(model: SimpleNet, optimizer, buffer: ReplayBuffer, path: str):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'buffer': list(buffer.buffer),
    }, path)


def load_checkpoint(model: SimpleNet, optimizer, buffer: ReplayBuffer, path: str):
    if os.path.exists(path):
        data = torch.load(path)
        model.load_state_dict(data['model'])
        optimizer.load_state_dict(data['optimizer'])
        buffer.buffer = deque(data['buffer'], maxlen=buffer.buffer.maxlen)


def main():
    model = SimpleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = ReplayBuffer(5000)

    load_checkpoint(model, optimizer, buffer, "checkpoint.pth")

    with open("training_log.csv", "a", newline="") as f:
        log_writer = csv.writer(f)
        if f.tell() == 0:
            log_writer.writerow(["avg_score", "avg_loss"])

        for iteration in range(2):
            examples, scores = self_play(model, episodes=2, simulations=50,
                                        temperature_moves=10, log_csv=None)
            buffer.add_examples(examples)

            losses = []
            for _ in range(5):
                batch_size = min(len(buffer.buffer), 32)
                batch = buffer.sample(batch_size)
                loss = train_step(model, optimizer, batch, batch_size)
                losses.append(loss)
            avg_loss = sum(losses) / len(losses)
            avg_score = sum(scores) / len(scores)
            log_writer.writerow([avg_score, avg_loss])
            print(f"Iteration {iteration} done, score {avg_score:.1f} loss {avg_loss:.4f}")
            save_checkpoint(model, optimizer, buffer, "candidate.pth")
            evaluate("checkpoint.pth", "candidate.pth", games=2)



if __name__ == "__main__":
    main()
