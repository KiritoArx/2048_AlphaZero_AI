#include "merge.hpp"
#include <iostream>

int main() {
    Board board{};
    printBoard(board);
    dropAndResolve(board, 2, 0);
    printBoard(board);
    dropAndResolve(board, 2, 0);
    printBoard(board);
    return 0;
}
