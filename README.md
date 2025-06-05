# 2048 AlphaZero AI

This repository contains a minimal C++ implementation of a dropping-tile merge board. The `merge.cpp` and `merge.hpp` files provide a small board representation and resolve logic inspired by 2048-style games. A simple `main.cpp` demonstrates dropping tiles and printing the board.

To compile the example program:

```bash
g++ -std=c++17 merge.cpp main.cpp -o merge_example
./merge_example
```

The project also includes a small Python framework implementing the same board logic. A simple neural network (`net.py`) and Monte Carlo Tree Search (`mcts.py`) can be used to run self‑play games with `train.py`.

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

Run a short training session (self-play plus updates with gradient clipping).
Each iteration saves the updated model to `candidate.pth`, evaluates it against
`checkpoint.pth`, and promotes the candidate if it scores higher. Average score
and training loss are appended to `training_log.csv`:

```bash
python train.py
```

An `evaluate.py` script can compare two saved models and promote the better
one:

```bash
python evaluate.py checkpoint.pth candidate.pth --games 5
```

This is a skeleton for experimenting with AlphaZero-style training on a 5×6 board.
