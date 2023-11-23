import tkinter as tk
from GymTTT import GymTTT
from QLearningAgent import QLearningAgent


class TicTacToeGUI:
    def __init__(self, agent):
        self.root = tk.Tk()
        self.root.title("Tic Tac Toe")
        self.gym = GymTTT(agent, verbose=True)

        self.buttons = [[None, None, None] for _ in range(3)]

        for i in range(3):
            for j in range(3):
                self.buttons[i][j] = tk.Button(
                    self.root,
                    text=" ",
                    font=("normal", 20),
                    command=lambda i=i, j=j: self.on_button_click(i, j),
                )
                self.buttons[i][j].grid(row=i, column=j, sticky="nsew")


        self.reset_button = tk.Button(
            self.root, text="Reset", command=self.reset_game
        )
        self.reset_button.grid(row=3, column=1)

        self.reset_game()

    def on_button_click(self, row, col):
        action = (row, col)
        print(f"Human Move: {action}")
        time_step = self.gym.step(action)

        self.update_board(time_step.observation)
        

    def reset_game(self):
        time_step = self.gym.reset()
        self.update_board(time_step.observation)

    def update_board(self, board):
        symbols = {0: " ", 1: "X", 2: "O"}

        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text=symbols[board[i][j]])



    def run(self):
        self.root.mainloop()


def main():
    # Carregar o agente treinado
    trained_agent = QLearningAgent(2, False)
    trained_agent.load("./agents/dql/difficult/hard_agent.pickle")

    # Crie um objeto TicTacToeGUI com o agente treinado
    app = TicTacToeGUI(trained_agent)

    # Execute o jogo
    app.run()

if __name__ == "__main__":
    main()
