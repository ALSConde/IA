import numpy as np
from DQLAgent import DQLAgent

tf.get_logger().setLevel('ERROR')

# Definindo o tabuleiro
class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9)  # 0 representa uma casa vazia

    def reset(self):
        self.board = np.zeros(9)  # Reinicia o tabuleiro

    def available_actions(self):
        return np.where(self.board == 0)[0]  # Retorna as casas vazias

    # Retorna a recompensa da jogada
    def step(self, action, player):
        if self.board[action] == 0:
            self.board[action] = player
            if self.check_win(player):
                return 1  # recompensa por vitória
            if len(self.available_actions()) == 0:
                return 0.5  # recompensa por empate
            return -0.1  # recompensa padrão por jogada
        return -1.5  # recompensa por jogada inválida

    # Jogada aleatória do oponente, retorna a recompensa para o agente caso a jogada do oponente resulte em vitória
    def random_play(self, player):
        available = self.available_actions()
        if len(available) == 0:
            return 0.5  # empate, recompensa para o agente é positiva
        move = np.random.choice(available)
        self.board[move] = player
        if self.check_win(player):
            return (
                -1
            )  # a jogada do agente resultou em uma vitória para o oponente, recompensa negativa
        return -0.1  # recompensa padrão por jogada do adversário, recompensa negativa

    # Jogada heurística do oponente, retorna a recompensa para o agente caso a jogada do oponente resulte em vitória
    def heuristic_play(self, player):
        opponent = -player
        center = 4
        corners = [0, 2, 6, 8]
        sides = [1, 3, 5, 7]

        # Jogada para vencer ou bloquear duas possíveis vitórias simultâneas
        winning_moves = []
        for i in self.available_actions():
            board_copy = self.board.copy()
            board_copy[i] = player
            if self.check_win(player, board_copy):
                winning_moves.append(i)

        if winning_moves:
            move = np.random.choice(winning_moves)
            self.board[move] = player
            return move, 1 if player == 1 else -1

        # Bloqueio de vitória do adversário
        for i in self.available_actions():
            board_copy = self.board.copy()
            board_copy[i] = opponent

            if self.check_win(opponent, board_copy):
                self.board[i] = player
                return i, -0.1

        # Tentar o centro
        if center in self.available_actions():
            self.board[center] = player
            return center, -0.1

        # Estratégia de canto oposto
        for i in corners:
            if self.board[i] == opponent and self.board[8 - i] == 0:
                self.board[8 - i] = player
                return (8 - i), -0.1

        # Tentar um canto
        available_corners = [
            corner for corner in corners if corner in self.available_actions()
        ]
        if available_corners:
            move = np.random.choice(available_corners)
            self.board[move] = player
            return move, -0.1

        # Tentar um lado
        available_sides = [side for side in sides if side in self.available_actions()]
        if available_sides:
            move = np.random.choice(available_sides)
            self.board[move] = player
            return move, -0.1

        # Jogada aleatória (isso não deve acontecer, mas é um backup)
        available_actions = self.available_actions()
        if available_actions.size > 0:
            move = np.random.choice(available_actions)
            return move, 0
        else:
            return 0, 0

    # verifica se o jogador venceu
    def check_win(self, player, board=None):
        # Se o tabuleiro não for passado como parâmetro, usa o tabuleiro atual
        if board is None:
            board = self.board

        # Condições de vitória
        win_conditions = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]

        # Verifica se alguma das condições de vitória foi satisfeita
        for condition in win_conditions:
            if (
                board[condition[0]]
                == board[condition[1]]
                == board[condition[2]]
                == player
            ):
                return True
        return False


if __name__ == "__main__":
    game = TicTacToe()
    state_size = 9
    action_size = 9
    agent = DQLAgent(state_size, action_size, game)
    episodes = 10
    batch_size = 32

    model_wins = 0
    heuristic_wins = 0
    draws = 0

    player_agent = 1
    opponent = -1

    for e in range(0, episodes):
        game.reset()
        state = np.reshape(game.board, [1, state_size])
        for time in range(9):
            # Verifica se ainda há ações disponíveis
            if not game.available_actions().size > 0:
                break

            # Jogada do agente
            action = agent.act(state)
            reward = game.step(action, player_agent)

            if reward == 1:
                model_wins += 1
            elif reward == -1:
                heuristic_wins += 1
            elif reward == 0:
                draws += 1

            if reward in [1, -1, 0]:  # Se o jogo terminou
                print(
                    f"episode: {e + 1}/{episodes}, score: {time}, e: {agent.epsilon:.2}, wins: {model_wins}, draws: {draws}, losses: {heuristic_wins}"
                )
                break

            # Jogada do oponente
            # reward_opponent = game.random_play(opponent) #Oponente aleatório
            _, reward_opponent = game.heuristic_play(opponent)  # Oponente heurístico
            if reward_opponent in [1, -1, 0]:
                if reward_opponent == 1:
                    heuristic_wins += 1
                elif reward_opponent == -1:
                    heuristic_wins += 1
                else:
                    draws += 1
                print(
                    f"episode: {e + 1}/{episodes}, score: {time}, e: {agent.epsilon:.2}, wins: {model_wins}, draws: {draws}, losses: {heuristic_wins}"
                )
                break

            next_state = np.reshape(game.board, [1, state_size])
            agent.remember(state, action, reward, next_state)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # Alterna o papel do agente e do oponente após cada episódio
        player_agent, opponent = opponent, player_agent

    agent.save("./dql_agent/difficult/easy")
    print("Treinamento concluído!")
    # print(f"Variável do agente: {agent.model.get_weights()}")
