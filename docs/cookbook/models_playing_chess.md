# Large language models playing chess

In this example we will make a quantized version of Mistral-7B play chess against itself. On its own the model easily generates invalid move, so we will give it a little help. At each step we will generate a regex that only matches valid move, and use it to help the model only generating valid moves.

## The chessboard

The game will be played on a standard checkboard. We will use the `chess` [library](https://github.com/niklasf/python-chess) to track the opponents' moves, and check that the moves are valid.

```python
import chess

board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
```

## The opponents

Mistral-7B quantized will be playing against itself:

```python
from outlines import models

board_state = models.transformers("TheBloke/Mistral-7B-OpenOrca-AWQ", device="cuda")
```

## A little help for the language model

To make sure Mistral-7B generates valid chess moves we will use Outline's regex-guided generation. We define a function that takes the current state of the board and returns a regex that matches all possible legal moves:

```python
import re

def legal_moves_regex(board):
    """Build a regex that only matches valid moves."""
    legal_moves = list(board.legal_moves)
    legal_modes_str = [board.san(move) for move in legal_moves]
    legal_modes_str = [re.sub(r"[+#]", "", move) for move in legal_modes_str]
    regex_pattern = "|".join(re.escape(move) for move in legal_modes_str)
    regex_pattern = f"{regex_pattern}"
    return regex_pattern
```

## Prompting the language model

The prompt corresponds to the current state of the board, so we start with:

```python
prompt = "Score: 1-0 WhiteElo: 1600 BlackElo: 1600 Timecontrol: 1800+0 Moves: 1."
```

We update the prompt at each step so it reflects the state of the board after the previous move.

## Let's play!


```python
from outlines import generate


turn_number = 0
while not board.is_game_over():
    regex_pattern = legal_moves_regex(board)
    guided = generate.regex(model, regex_pattern)(board_state)
    move = board.parse_san(guided)

    if turn_number % 2 == 0 :  # It's White's turn
        board_state += board.san(move) + " "
    else:
        board_state += board.san(move) + " " + str(turn_number) + "."

    turn_number += 1

    board.push(move)

    print(board_state)
```

It turns out Mistal-7B (quantized) is not very good at playing chess: the game systematically ends because of the threefold repetition rule.


*This example was originally authored by [@903124S](https://x.com/903124S) in [this gist](https://gist.github.com/903124/cfbefa24da95e2316e0d5e8ef8ed360d).*
