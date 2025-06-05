from collections import deque
import random
import math

import torch
import torch.nn.functional as F

from game import Game, game_over, COLUMN_COUNT
from mcts import MCTS, Node
from net import SimpleNet, encode_game


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


def self_play(model, episodes: int = 1, max_moves: int = 50):
    """Run self-play games and return training examples.

    Each example is a tuple (state, policy, value) where `state` is a cloned
    ``Game`` instance, `policy` is the MCTS visit distribution, and ``value`` is
    the final game score scaled to ``[-1, 1]``.
    """
    examples = []
    for _ in range(episodes):
        game = Game()
        root = Node(game)
        mcts = MCTS(model, simulations=25)
        trajectory = []
        move = 0
        done = False
        while not done and move < max_moves:
            mcts.run(root)
            visits = {a: child.visit_count for a, child in root.children.items()}
            total = sum(visits.values()) + 1e-8
            policy = [visits.get(a, 0) / total for a in range(COLUMN_COUNT)]

            trajectory.append((game.clone(), policy))

            action = mcts.select_action(root)
            reward, done = game.step(action)
            root = Node(game)
            move += 1
        # use final score scaled to [-1, 1] as value target
        value = math.tanh(game.score / 1000.0)
        for state, pol in trajectory:
            examples.append((state, pol, value))
    return examples


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


def main():
    model = SimpleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = ReplayBuffer(5000)

    for iteration in range(2):
        examples = self_play(model, episodes=2)
        buffer.add_examples(examples)

        for _ in range(5):
            batch_size = min(len(buffer.buffer), 32)
            batch = buffer.sample(batch_size)
            loss = train_step(model, optimizer, batch, batch_size)
        print(f"Iteration {iteration} done, buffer size {len(buffer.buffer)}")


if __name__ == "__main__":
    main()
