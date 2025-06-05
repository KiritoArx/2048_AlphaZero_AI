from game import Game, game_over
from mcts import MCTS, Node
from net import SimpleNet


def self_play(model, episodes=1, max_moves=50):
    for _ in range(episodes):
        game = Game()
        root = Node(game)
        mcts = MCTS(model, simulations=25)
        move = 0
        while not game_over(game.board) and move < max_moves:
            mcts.run(root)
            action = mcts.select_action(root)
            reward, done = game.step(action)
            print(f"Move {move}: drop in {action} reward {reward}")
            game.render()
            root = Node(game)
            move += 1
            if done:
                break
        print("Game over. Score:", game.score)

if __name__ == "__main__":
    model = SimpleNet()
    self_play(model)
