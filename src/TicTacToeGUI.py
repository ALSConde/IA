import tkinter as tk
from tkinter import messagebox
from DQLAgent import DQLAgent
from DDQLAgent import DDQLAgent
from TicTacToe import TicTacToe

tf.get_logger().setLevel('ERROR')

class TicTacToeGUI:
    def __init__(self):
        self.game = TicTacToe()

        self.root = tk.Tk()  # Inicialize isso primeiro!
        self.root.title("Jogo da Velha - IA")

        self.agent_type = tk.StringVar()  # Em seguida, inicialize as variáveis tkinter
        self.agent_type.set("Heurístico")

        self.buttons = [
            tk.Button(
                self.root,
                text="",
                width=20,
                height=3,
                command=lambda i=i: self.player_move(i),
            )
            for i in range(9)
        ]
        for i, btn in enumerate(self.buttons):
            row, col = divmod(i, 3)
            btn.grid(row=row, column=col)

        self.label = tk.Label(self.root, text="Escolha o agente:")
        self.label.grid(row=3, column=0, columnspan=3)

        self.choices = ["DQL", "DDQL", "Heurístico"]
        self.dropdown = tk.OptionMenu(self.root, self.agent_type, *self.choices)
        self.dropdown.grid(row=4, column=0, columnspan=3)

        # Adicionando um dropdown para selecionar a dificuldade
        self.difficulty_label = tk.Label(self.root, text="Dificuldade:")
        self.difficulty_label.grid(row=6, column=0, columnspan=3)
        self.difficulty_var = tk.StringVar()
        self.difficulty_var.set("Fácil")
        self.difficulties = ["Fácil", "Normal", "Difícil"]
        self.difficulty_dropdown = tk.OptionMenu(
            self.root, self.difficulty_var, *self.difficulties
        )
        self.difficulty_dropdown.grid(row=7, column=0, columnspan=3)

        # Atualiza a visibilidade da opção de dificuldade dependendo da seleção do tipo de agente
        self.agent_type.trace_add("write", self.update_difficulty_visibility)
        self.update_difficulty_visibility()

        self.restart_button = tk.Button(
            self.root, text="Reiniciar", command=self.restart_game
        )
        self.restart_button.grid(row=5, column=0, columnspan=3)

    def update_difficulty_visibility(self, *args):
        if self.agent_type.get() == "Heurístico":
            self.difficulty_label.grid_remove()
            self.difficulty_dropdown.grid_remove()
        else:
            self.difficulty_label.grid()
            self.difficulty_dropdown.grid()

    def player_move(self, index):
        if self.game.board[index] == 0:
            self.buttons[index].config(text="X")
            self.game.board[index] = 1

            if not self.game.check_win(1):
                self.agent_move()
            elif len(self.game.available_actions()) == 0:
                messagebox.showinfo("Resultado", "Empate!")
                self.restart_game()
            else:
                messagebox.showinfo("Resultado", "Você venceu!")
                self.restart_game()

    def agent_move(self):
        # Adicionando feedback visual
        # self.feedback_label.config(text="Pensando...")
        self.root.update_idletasks()  # força a atualização da GUI

        difficulty = self.difficulty_var.get()

        if self.agent_type.get() == "DQL":
            agent = DQLAgent(9, 9, self.game, epsilon=0.0)
            if difficulty == "Fácil":
                agent.load("./dql_agent/difficult/easy")
            elif difficulty == "Normal":
                agent.load("./dql_agent/difficult/normal")
            else:
                agent.load("./dql_agent/difficult/hard")
            action = agent.act(self.game.board)
            self.game.board[action] = -1
        elif self.agent_type.get() == "DDQL":
            agent = DDQLAgent(9, 9, self.game)
            if difficulty == "Fácil":
                agent.load("./path_to_ddql_model_10/")
            elif difficulty == "Normal":
                agent.load("./path_to_ddql_model_100/")
            else:
                agent.load("./path_to_ddql_model_150/")
            action = agent.act(self.game.board)
            self.game.board[action] = -1
        else:
            action, _ = self.game.heuristic_play(-1)

        # self.feedback_label.config(text="")  # remove o feedback
        self.buttons[action].config(text="O")

        if self.game.check_win(-1):
            messagebox.showinfo("Resultado", "IA venceu!")
            self.restart_game()
        elif len(self.game.available_actions()) == 0:
            messagebox.showinfo("Resultado", "Empate!")
            self.restart_game()

    def restart_game(self):
        self.game.reset()
        for btn in self.buttons:
            btn.config(text="")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui_game = TicTacToeGUI()
    gui_game.run()
