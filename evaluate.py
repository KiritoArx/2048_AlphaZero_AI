import argparse
import shutil
import statistics
import torch
from game import Game
from mcts import MCTS, Node
from net import SimpleNet


def play_game(model: SimpleNet, simulations: int = 50, max_moves: int = 50) -> int:
    game = Game()
    root = Node(game)
    mcts = MCTS(model, simulations=simulations, batch_size=8)
    move = 0
    done = False
    while not done and move < max_moves:
        mcts.run(root)
        action = mcts.select_action(root)
        _, done = game.step(action)
        root = Node(game)
        move += 1
    return game.score


def load_model(path: str) -> SimpleNet:
    model = SimpleNet()
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    model.load_state_dict(sd)
    model.eval()
    return model


def evaluate(current_path: str, candidate_path: str, games: int) -> None:
    cur = load_model(current_path)
    cand = load_model(candidate_path)
    scores_cur = [play_game(cur) for _ in range(games)]
    scores_cand = [play_game(cand) for _ in range(games)]
    avg_cur = statistics.mean(scores_cur)
    avg_cand = statistics.mean(scores_cand)
    print(f"Current avg: {avg_cur:.1f} Candidate avg: {avg_cand:.1f}")
    if avg_cand > avg_cur:
        shutil.copy(candidate_path, current_path)
        print("Candidate promoted to current model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("current")
    parser.add_argument("candidate")
    parser.add_argument("--games", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.current, args.candidate, args.games)
