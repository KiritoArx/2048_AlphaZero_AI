import math
import random
from typing import Dict, Optional
from game import Game

class Node:
    def __init__(self, game: Game, parent: Optional['Node']=None, prior: float=0.0):
        self.game = game
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, Node] = {}

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, simulations: int = 50, c_puct: float = 1.0):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct

    def run(self, root: Node):
        for _ in range(self.simulations):
            node = root
            search_path = [node]
            # Selection
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)
            # Expansion
            value = self.expand(node)
            # Backpropagation
            for n in reversed(search_path):
                n.value_sum += value
                n.visit_count += 1
                value = -value

    def select_child(self, node: Node):
        best_score = -1e9
        best_action = None
        best_child = None
        for action, child in node.children.items():
            ucb = child.value() + self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child
        return best_action, best_child

    def expand(self, node: Node) -> float:
        game = node.game
        if game.valid_actions() == []:
            return 0.0
        policy, value = self.model.predict(game)
        total = sum(policy[a] for a in game.valid_actions()) + 1e-8
        for action in game.valid_actions():
            p = policy[action] / total
            g = game.clone()
            reward, done = g.step(action)
            if done:
                child = Node(g, node, p)
                child.value_sum = reward
                child.visit_count = 1
            else:
                child = Node(g, node, p)
            node.children[action] = child
        return value

    def select_action(self, root: Node) -> int:
        counts = {action: child.visit_count for action, child in root.children.items()}
        return max(counts, key=counts.get)
