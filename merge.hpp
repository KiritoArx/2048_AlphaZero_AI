#pragma once
#include <array>
#include <vector>

constexpr int COLUMN_COUNT = 5;
constexpr int MAX_HEIGHT = 6;

/**
 * Board representation. Each column is a vector of integers where index 0 is the
 * bottom of the board. The board has fixed COLUMN_COUNT columns.
 */
using Board = std::array<std::vector<int>, COLUMN_COUNT>;

void printBoard(const Board& board);

/// Drop a tile with the given value into the specified column and resolve merges.
void dropAndResolve(Board& board, int value, int col);

/// Returns true if any column has reached MAX_HEIGHT.
bool gameOver(const Board& board);
