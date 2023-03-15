import numpy as np
import random

class TicTacToe:
    def __init__(self, model=None):
        self.board = [' '] * 9
        self.current_player = '1'
        self.model = model
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.winner = None

    def display(self):
        for row in self.board:
            print("|".join(row))
            print("-" * 5)

    def make_move(self, position, player):
        x, y = position
        if self.board[x][y] == ' ':
            self.board[x][y] = player
            if self.check_winner(position, player):
                self.winner = player
            return True
        return False

    def check_winner(self, position, player):
        x, y = position
        win = (
            all(self.board[x][col] == player for col in range(3)) or
            all(self.board[row][y] == player for row in range(3)) or
            (x == y and all(self.board[i][i] == player for i in range(3))) or
            (x + y == 2 and all(self.board[i][2 - i] == player for i in range(3)))
        )
        return win

    def game_over(self):
        return (
            self.winner is not None or
            all(all(cell != ' ' for cell in row) for row in self.board)
        )

    def legal_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves
    
    def evaluate(self, player):
        if self.winner is None:
            return 0
        elif self.winner == player:
            return 1
        else:
            return -1
    
    def get_opponent(self, player):
        return '1' if player == '2' else '2'

    def undo_move(self, position):
        x, y = position
        if self.board[x][y] != ' ':
            self.board[x][y] = ' '
            self.winner = None

    def get_valid_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves


class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def state_to_key(self, state):
        return tuple(state.flatten())

    def get_q_value(self, state, action):
        key = self.state_to_key(state)
        return self.q_table.get((key, action), 0.0)

    def update_q_value(self, state, action, value):
        key = self.state_to_key(state)
        self.q_table[(key, action)] = value

    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        q_values = [self.get_q_value(state, action) for action in available_actions]
        max_q = max(q_values)

        best_actions = [action for action, q_value in zip(available_actions, q_values) if q_value == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_available_actions):
        q_value = self.get_q_value(state, action)
        max_next_q_value = max([self.get_q_value(next_state, next_action) for next_action in next_available_actions]) if next_available_actions else 0
        updated_q_value = q_value + self.alpha * (reward + self.gamma * max_next_q_value - q_value)
        self.update_q_value(state, action, updated_q_value)

def available_actions(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]

def train_agents(agent1, agent2, env, episodes=50000):
    for _ in range(episodes):
        play_game(agent1, agent2, env)

def test_agents(agent1, agent2, env, games=1000):
    agent1.epsilon = 0
    agent2.epsilon = 0
    results = {'1': 0, '2': 0, 'draw': 0}
    for _ in range(games):
        play_game(agent1, agent2, env, train=False)
        winner = None

    if env.check_winner():
        winner = str(env.player)
    else:
        winner = 'draw'

    results[winner] += 1

    return results

def get_best_move(game, player):
    optimal_score = float('-inf')
    optimal_move = None

    for move in game.legal_moves():
        game.make_move(move, player)
        score = minimax_ab(game, game.get_opponent(player), False, alpha=float('-inf'), beta=float('inf'))
        game.undo_move(move)

        if score > optimal_score:
            optimal_score = score
            optimal_move = move

    return optimal_move

def minimax_ab(game, player, maximizing_player, alpha, beta):
    if game.game_over():
        return game.evaluate(player)

    if maximizing_player:
        max_eval = float('-inf')
        for move in game.legal_moves():
            game.make_move(move, player)
            eval = minimax_ab(game, game.get_opponent(player), False, alpha, beta)
            game.undo_move(move)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval

    else:
        min_eval = float('inf')
        for move in game.legal_moves():
            game.make_move(move, player)
            eval = minimax_ab(game, game.get_opponent(player), True, alpha, beta)
            game.undo_move(move)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def play_game(game, player1, player2, show_steps=False):
    current_player = player1

    while not game.game_over():
        if show_steps:
            game.display()

        move = get_best_move(game, current_player) if current_player != 'human' else get_human_move(game)
        game.make_move(move, current_player)

        if game.winner:
            if show_steps:
                game.display()
                print(f"Player {current_player} wins!")
            return current_player
        elif game.game_over():
            if show_steps:
                game.display()
                print("It's a draw!")
            return 'draw'

        current_player = player2 if current_player == player1 else player1

def get_human_move(game):
    valid_moves = game.get_valid_moves()
    move = (-1, -1)
    while move not in valid_moves:
        try:
            index = int(input("Enter your move (0-8): "))
            if index < 0 or index > 8:
                raise ValueError("Invalid input. Please enter a number between 0 and 8.")
            move = (index // 3, index % 3)
            if move not in valid_moves:
                print("Invalid move. Please choose an empty space.")
        except ValueError as e:
            print(e)
    return move

def main():
    # Train the model
    num_games = 1000
    results = {'1': 0, '2': 0, 'draw': 0}
    for _ in range(num_games):
        game = TicTacToe()
        result = play_game(game, '1', '2')
        results[result] += 1

    print("Results after", num_games, "games:", results)

    while True:
        choice = input("Enter 'human' to play against the model, 'watch' to watch the model play, or 'quit' to exit: ").lower()

        if choice == 'human':
            game = TicTacToe()
            play_game(game, '1', 'human', show_steps=True)
        elif choice == 'watch':
            game = TicTacToe()
            play_game(game, '1', '2', show_steps=True)
        elif choice == 'quit':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()



