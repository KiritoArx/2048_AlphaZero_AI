import math
import random
from typing import Dict, Optional, List
from game import Game, game_over

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
    def __init__(self, model, simulations: int = 50, c_puct: float = 1.0,
                 batch_size: int = 1, value_weight: float = 1.0):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.value_weight = value_weight

    def run(self, root: Node):
        leaves: List[Node] = []
        paths: List[List[Node]] = []
        for _ in range(self.simulations):
            node = root
            search_path = [node]
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)
            leaves.append(node)
            paths.append(search_path)
            if len(leaves) >= self.batch_size:
                self._evaluate_and_backpropagate(leaves, paths)
                leaves, paths = [], []
        if leaves:
            self._evaluate_and_backpropagate(leaves, paths)

    def select_child(self, node: Node):
        best_score = -1e9
        best_action = None
        best_child = None
        for action, child in node.children.items():
            ucb = (self.value_weight * child.value() +
                   self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count))
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

    def select_action(self, root: Node, temperature: float = 0.0) -> int:
        counts = {action: child.visit_count for action, child in root.children.items()}
        actions = list(counts.keys())
        visits = [counts[a] for a in actions]
        if temperature <= 0:
            return actions[int(visits.index(max(visits)))]
        visits = [v ** (1 / temperature) for v in visits]
        total = sum(visits)
        probs = [v / total for v in visits]
        return random.choices(actions, probs)[0]

    def _evaluate_and_backpropagate(self, leaves: List[Node], paths: List[List[Node]]):
        games = [leaf.game for leaf in leaves]
        policies, values = self.model.predict_batch(games)
        for leaf, path, policy, value in zip(leaves, paths, policies, values):
            if not leaf.expanded() and not game_over(leaf.game.board):
                total = sum(policy[a] for a in leaf.game.valid_actions()) + 1e-8
                for action in leaf.game.valid_actions():
                    p = policy[action] / total
                    g = leaf.game.clone()
                    reward, done = g.step(action)
                    child = Node(g, leaf, p)
                    if done:
                        child.value_sum = reward
                        child.visit_count = 1
                    leaf.children[action] = child
            val = value
            for n in reversed(path):
                n.value_sum += val
                n.visit_count += 1
                val = -val
