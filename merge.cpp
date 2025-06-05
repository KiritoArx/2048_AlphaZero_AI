#include "merge.hpp"
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>

/* Pretty-printer (tight ASCII) */
void printBoard(const Board& board)
{
    for (int row = MAX_HEIGHT - 1; row >= 0; --row) {
        for (int col = 0; col < COLUMN_COUNT; ++col) {
            if (row < static_cast<int>(board[col].size()))
                std::cout << std::setw(6) << board[col][row];
            else
                std::cout << std::setw(6) << '.';
        }
        std::cout << '\n';
    }
    std::cout << std::string(COLUMN_COUNT * 6, '-') << '\n';
    for (int col = 0; col < COLUMN_COUNT; ++col)
        std::cout << std::setw(6) << ("C" + std::to_string(col));
    std::cout << "\n\n";
}

/* Drop-and-resolve */
void dropAndResolve(Board& board, int value, int col)
{
    board[col].push_back(value);
    int row = static_cast<int>(board[col].size()) - 1;

    const int landingCol = col;
    const int landingRow = row;

    std::deque<std::pair<int,int>> fresh{{col,row}};

    while (!fresh.empty()) {
        auto [c,r] = fresh.front();
        fresh.pop_front();

        if (r >= static_cast<int>(board[c].size())) continue;
        int v = board[c][r];

        /* vertical-down */
        if (r > 0 && board[c][r-1] == v) {
            board[c][r-1] *= 2;
            board[c].erase(board[c].begin() + r);
            fresh.push_back({c, r-1});                    // << changed
            if (r < static_cast<int>(board[c].size()))
                fresh.push_back({c, r});
            continue;
        }

        /* vertical-up */
        if (r + 1 < static_cast<int>(board[c].size()) &&
            board[c][r+1] == v)
        {
            board[c][r] *= 2;
            board[c].erase(board[c].begin() + r + 1);
            fresh.push_back({c, r});                      // << changed
            if (r + 1 < static_cast<int>(board[c].size()))
                fresh.push_back({c, r + 1});
            continue;
        }

        /* horizontal */
        for (int dc : {-1, 1}) {
            int nc = c + dc;
            if (nc < 0 || nc >= COLUMN_COUNT) continue;
            if (r < static_cast<int>(board[nc].size()) &&
                board[nc][r] == v)
            {
                bool keepRight = (r == landingRow && nc == landingCol);
                int dst = keepRight ? nc : c;
                int src = keepRight ? c  : nc;

                board[dst][r] *= 2;
                board[src].erase(board[src].begin() + r);

                fresh.push_back({dst, r});                // << changed
                if (r < static_cast<int>(board[src].size()))
                    fresh.push_back({src, r});
                break;
            }
        }
    }
}

bool gameOver(const Board& board)
{
    for (const auto& col : board)
        if (static_cast<int>(col.size()) >= MAX_HEIGHT)
            return true;
    return false;
}
