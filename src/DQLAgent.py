import numpy as np
import random
import tensorflow as tf
from collections import deque


class DQLAgent:
    def __init__(self, state_size: int, action_size: int, game, epsilon : int=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.learning_rate = 0.1
        self.model = self._build_model()
        self.game = game

    def _build_model(self):
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation="relu")
        )
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(self.action_size, activation="linear"))

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate, decay_steps=10000, decay_rate=0.9
        )

        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        )
        return model

    def filter_valid_actions(self, predictions):
        valid_actions = self.game.available_actions()
        valid_predictions = [predictions[i] for i in valid_actions]
        return valid_actions[np.argmax(valid_predictions)]

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.game.available_actions())

        act_values = self.model.predict(state.reshape(1, -1))[0]

        return self.filter_valid_actions(act_values)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            if reward not in [-0.1, -1]:  # final de jogo ou jogada inválida
                target = reward
            else:
                next_actions = self.model.predict(next_state)[0]
                if reward == -0.1:  # turno do agente
                    target = reward + self.gamma * np.max(next_actions)
                else:  # turno do adversário
                    target = reward + self.gamma * np.min(next_actions)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.train_on_batch(state, target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, fp):
        self.model.save(fp, overwrite=True, save_format="tf")

    def load(self, fp):
        self.model = tf.keras.models.load_model(fp)
